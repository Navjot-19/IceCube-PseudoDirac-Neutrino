import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import healpy as hp

# Settings
mpl.rcParams['font.family'] ='serif'
mpl.rcParams['text.usetex']= False
mpl.rcParams['font.size']=13

#===============================================================================================================
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "../resources/1_logpVal_skymap.txt")

pval = np.genfromtxt(file_path, skip_header=True).T[2]
print(pval.shape)

#===============================================================================================================
#Plot
#rot=(180,0,0)                                                                                 ?
#coord=["G"], give G=galactic coordinates, C=ecliptic coordinates, E=equatorial
#graticule=True gives lines or grid displays on sky map
#unit=r"text001"
#cb_orientation="horizontl" gives color bar orientation
#projection_type='hammer' gives hammer projection                                              ?
#min,max sets range of colorbar
#cmap='bone_r' gives reverse bone- grey map                                                    ?
#flip the map- east left, west right                                                           ?
#longitud_grid_spacing= 45 degree, latitute_grid_spacing=30 degree
#graticule_color='red' or 'silver'
hp.projview(pval,
            rot=(180,0,0),
            graticule=True, 
            coord=["G"],            
            unit=r"$ log_{10}(p_{local})$",
            cb_orientation='horizontal',
            projection_type='hammer',                                    #hammer or mollweide
            min=0, max=7,
            cmap='bone_r',                               #bone- darker  #bone_r #jet   #viridis
            flip='astro',                                #flip=astro increases from right to left
            longitude_grid_spacing=15,
            latitude_grid_spacing=30,
            graticule_color='silver',
            override_plot_properties={'cbar_label_pad':15, 'cbar_pad':0.15, "cbar_tick_direction":"out"},
            fontsize={'cbar_label':20, 'cbar_tick_label':10, 'xtick_label':16, 'ytick_label':16},
            cbar_ticks={1,2,3,4,5,6,7},
            show_tickmarkers=True,
            xtick_label_color='k', 
            margins=()
            ) 
#===============================================================================================================

ax=plt.gca()
ax.set_yticklabels([r"-60",r"-30",r"0",r"30",r"60"], fontsize=10)
ax.text(-0.05,0.4,"0h",horizontalalignment="left", verticalalignment="bottom",     #transform parameter choose the relative to area
        transform=ax.transAxes,fontsize=16)
ax.text(0.5,0.4,"12h",horizontalalignment="center", verticalalignment="bottom",
        transform=ax.transAxes,fontsize=16)
ax.text(1.05,0.4,"24h",horizontalalignment="center", verticalalignment="bottom",
        transform=ax.transAxes,fontsize=16)

#===============================================================================================================
#3sources
x,y =np.radians(180-4.0667e+01), np.radians(-6.9e-03)
ax.scatter(x,y,marker='o',s=100,
           facecolors='none',edgecolors='k')
ax.text(x+np.radians(5),y+np.radians(7),r"NGC 1068",
        verticalalignment='center',
        horizontalalignment='center')

x, y = np.radians(180-216.76), np.radians(23.80)
ax.scatter(x,y,marker='o',s=100,
           facecolors='none',edgecolors='k')
ax.text(x+np.radians(5),y+np.radians(7),r"PKS 1424+240",
        verticalalignment='center',
        horizontalalignment='center')


x, y = np.radians(180-77.36), np.radians(5.70)
ax.scatter(x,y,marker='o',s=100,
           facecolors='none',edgecolor='k')
ax.text(x+np.radians(15),y+np.radians(17),r"TXS 0506+056",
        verticalalignment='center',
        horizontalalignment='center')

ax.set_xlabel("Right ascension[deg]", labelpad=10,fontsize=18)
ax.set_ylabel("Declination[deg]", labelpad=15, fontsize=18)

#===============================================================================================================
plt.show()
