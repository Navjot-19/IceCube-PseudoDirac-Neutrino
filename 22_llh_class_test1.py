import os
import sys
try:
    import photospline as psp
except:
    try:
        sys.path.append(os.environ['photospline_path'])
        import photospline as psp
    except:
        print(f'No photospline found.')
        exit()

#===============================================================================================================

from scipy.optimize import minimize, fmin_l_bfgs_b
import photospline as psp
import pickle
import numpy as np
import math
import copy

def angular_distance(ra_1,dec_1,ra_2,dec_2):
    delta_dec=np.abs(dec_1-dec_2)
    delta_ra=np.abs(ra_1-ra_2)
    sin2_x_by_2=(np.sin(delta_dec/2))**2+ np.cos(dec_1)*np.cos(dec_2)*np.sin(delta_ra/2)**2
    return 2*np.arcsin(np.sqrt(sin2_x_by_2))

class likelihood():

    def __init__(self, src_ra,src_dec):
        self.src_dec=np.radians(src_dec)
        self.src_ra=np.radians(src_ra)
        self.sin_src_dec=np.sin(self.src_dec)
        self.samples=None
         #load KDE
        kde_info=dict()
        this_folder=os.path.dirname(os.path.abspath(__file__))
        kde_basepath=os.path.join(this_folder, '../kdes/')

        kde_info['sig_spatial_pdf']=os.path.join(kde_basepath,'sig_E_psi_photospline_v006_4D.fits')
        kde_info['sig_spatial_info']=os.path.join(kde_basepath,'sig_E_psi_photospline_v006_4D_info.pkl')

        kde_info['sig_energy_pdf']=os.path.join(kde_basepath,'E_dec_photospline_v006_3D.fits')
        kde_info['sig_energy_info']=os.path.join(kde_basepath,'E_dec_photospline_v006_3D_info.pkl')

        kde_info['bkg_pdf']=os.path.join(kde_basepath,'bg_2d_photospline.fits')
        kde_info['bkg_info']=os.path.join(kde_basepath,'bg_2d_info.pkl')

        keys_info=['sig_spatial_info','sig_energy_info','bkg_info']
        keys_vars=['sig_spatial_vars','sig_energy_vars','bkg_vars']

        for key_info,key_var in zip(keys_info,keys_vars):
            data=pickle.load(open(kde_info[key_info],'rb'))
            kde_info[key_var]=data['vars']

        self.spatial_pdf=psp.SplineTable(kde_info['sig_spatial_pdf'])
        self.energy_pdf=psp.SplineTable(kde_info['sig_energy_pdf'])
        self.bkg_pdf=psp.SplineTable(kde_info['bkg_pdf'])
        
    def _eval_kde_bkg(self):
        return self.bkg_pdf.evaluate_simple([self.samples['log10_energy'],
                                             self.samples['sin_dec']])
    
    def _eval_kde_spatial(self,gamma,mode=0):
        return self.spatial_pdf.evaluate_simple([self.samples['log10_ang_err'],
                                                 self.samples['log10_energy'],
                                                 self.samples['log10_psi'],
                                                 np.full(self.Nprime,gamma)
                                                 ],
                                                 mode)
    
    def _eval_kde_energy(self, gamma,mode=0):
        return self.energy_pdf.evaluate_simple([self.samples['log10_energy'],
                                                np.full(self.Nprime,self.sin_src_dec),
                                                np.full(self.Nprime,gamma)],
                                                mode)
    
    def _eval_kde_energy_dgamma(self,gamma):
        return self._eval_kde_energy(gamma,mode=4)
    
    def _eval_kde_spatial_dgamma(self,gamma):
        return self._eval_kde_spatial(gamma,mode=8)
    
    def _dlogl_dns(self,ns,sb_ratio,func_vals):
        dns=np.sum((sb_ratio-1)/func_vals)
        alpha=ns/self.N
        dns += self.DeltaN *(-1)/(1-alpha)
        return 1/self.N*dns
    
    def set_src_pos(self,src_ra,src_dec):
        self.src_ra=np.radians(src_ra)
        self.src_dec=np.radians(src_dec)
        self._select_and_prepare_data()
        return
    
    def _dlogl_dgamma(self,ns, gamma, func_vals, spdfs,spatial_norm,epdfs):
        dgamma=1/(func_vals*self.bkg_pdf_values)
        product_rule=spdfs*self._eval_kde_energy_dgamma(gamma)+epdfs*self._eval_kde_spatial_dgamma(gamma)/spatial_norm
        dgamma=np.sum(dgamma*product_rule)
        return ns/self.N*dgamma
    
    def set_data(self,sample):
        self.samples_full=sample
        self.samples_full['log10_energy']=self.samples_full['logE']
        self.samples_full['sin_dec']=np.sin(self.samples_full['dec'])
        self.samples_full['log10_ang_err']=np.log10(self.samples_full['angErr'])
        self.N=665293
        self._select_and_prepare_data()
        return
    
    def _select_and_prepare_data(self,eps=1.e-20,box_size=15):
                
        psi=angular_distance(self.samples_full['ra'],self.samples_full['dec'],          
                             self.src_ra,self.src_dec )
        self.samples_full['psi']=psi
        self.samples=self.samples_full.loc[self.samples['psi']< np.radians(box_size)].copy()
        self.samples['log10_psi']=np.log10(self.samples['psi'])
        self.samples['sin_psi']=np.sin(self.samples['psi'])
        self.samples.sort_values('angErr',inplace=True)
        self.bkg_pdf_values=self._eval_kde_bkg()
        idx=np.where(self.bkg_pdf_values<0)[0]
        self.bkg_pdf_values[idx]=eps
        self.Nprime=len(self.samples)
        self.DeltaN=self.N-self.Nprime
        self.eps=eps
    
    

    def fit(self):
        ftot=1.e-6
        pgtot=1.e-6
        factr=ftot/np.finfo(float).eps

        def fm(x):
            fval,grad=self.logl(x[0],x[1],
                                eval_gradient=True)
            fval *=-2.0
            grad *=-2.0
            return fval,grad
        
        
        # run minimization

        result, fval, warn=fmin_l_bfgs_b(fm,(10,2.5),
                                         fprime=None,
                                         approx_grad=False,
                                         bounds=[(0,1000),(1.0,4.0)],
                                         pgtot=pgtot,
                                         factr=factr
                                         )
        return result,-fval
    
    def logl(self, ns, gamma, eval_gradient=False, fixed_gamma=False):
        if (ns< 0 or gamma<0):
            if (eval_gradient):
                return -1/ self.eps,np.asarray([-0.5,0.5])
            else:
                return -1/self.eps
        
        #evaluate pdfs

        spdfs=self._eval_kde_spatial(gamma)
        spatial_norm=np.log(10)*self.samples['psi']*self.samples['sin_psi']
        spdfs/=spatial_norm

        epdfs=self._eval_kde_energy(gamma)
        sb_ratio=spdfs*epdfs/self.bkg_pdf_values

        llh_vals=ns/self.N*(sb_ratio-1)+1

        logl=np.sum(np.log(llh_vals))

        logl+=self.DeltaN*np.log(1-ns/self.N)

        if(eval_gradient ) & (not fixed_gamma):
            dns=self._dlogl_dns(ns,sb_ratio, llh_vals)
            dgamma=self._dlogl_dgamma(ns,gamma, llh_vals,spdfs,spatial_norm, epdfs)
            return(logl,np.asarray([dns,dgamma]))
        
        elif (eval_gradient)&(fixed_gamma):
            dns=self._dlogl_dns(ns, sb_ratio, llh_vals)
            return(logl, np.asarray([dns]))
        
        else:
            return logl
             
likelihood_new=likelihood(src_ra=4.0667e+01,src_dec=-6.9e-03) 

spatial_pdf=likelihood_new.spatial_pdf
energy_pdf=likelihood_new.energy_pdf
bkg_pdf=likelihood_new.bkg_pdf

print(spatial_pdf,energy_pdf,bkg_pdf) 
            