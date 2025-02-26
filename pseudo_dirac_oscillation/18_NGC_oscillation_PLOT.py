import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import quad,dblquad

#===============================================================================================================
# Define constants

theta_12= 0.59
theta_23= 0.86
theta_13= 0.15
dirac_cp_phase=0
Delta_atm = 2.55e-3 * 1e-18  # (GeV^2)
Delta_sol = 7.50e-5 * 1e-18  # (GeV^2)


#===============================================================================================================
# Define U_mixing_matrix --- (ArXiv-  )
  
def R23(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),m.sin(theta)],
                   [ 0, -m.sin(theta), m.cos(theta)]])
  
def R13(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def R12(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def Uphase(dirac_cp_phase):
  return np.matrix([[ 1,         0,               0 ],
                   [ 0, np.exp(1j*dirac_cp_phase) , 0 ],
                   [ 0      , 0    ,np.exp(-1j *dirac_cp_phase)]])


U = R23(theta_23) @ Uphase(dirac_cp_phase)@ R13(theta_13) @ R12(theta_12)
print(U)

#===============================================================================================================
# Effective length -- due to redshifted neutrino energy            (ArXiv- 2212.00737)

def hubble_parameter(z, H0, Omega_m, Omega_Lambda):

    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda + (1 - Omega_m - Omega_Lambda) * (1 + z)**2)

def integrand(z, H0, Omega_m, Omega_Lambda):

    return 1 / (hubble_parameter(z, H0, Omega_m, Omega_Lambda) * (1 + z)**2)

def compute_Leff(z_min, z_max, H0, Omega_m, Omega_Lambda):

    Leff, _ = quad(integrand, z_min, z_max, args=(H0, Omega_m, Omega_Lambda))
    c = 3e5 
    Mpc_to_km = 3.0857e19  
    L_km = Leff * (c / H0) * Mpc_to_km
    return L_km


#===============================================================================================================
# Best fit from Planck Collaboration  (ArXiv- 1807.06209)

H0 = 67.36                                # Hubble constant in km/s/Mpc
Omega_m = 0.31                            # Matter density parameter
Omega_Lambda = 0.68                       # Cosmological constant density parameter
z_min = 0                                 # Start of redshift range
z_max = 0.0038                            # redshift for NGC 1068


L = compute_Leff(z_min, z_max, H0, Omega_m, Omega_Lambda)
print(f"L = {L:.2f} (in km)")


#===============================================================================================================
# Neutrino Oscillation Probability --- (ArXiv- 2212.00737 )

def oscillation_probability(alpha, beta, E_nu, dm2, L,U):
    P=0
    for j in range(3):
       P+=0.5*(abs(U[beta,j])**2)*(abs(U[alpha,j])**2)*(1+m.cos(dm2/2 * L / E_nu))  
    return P


#===============================================================================================================
# Plot


E_nu = np.logspace(-0.8, 2, 300)            # GeV  Start=10^-1, Stop=10^2, 100 points  (include end points)
dm2 = 10**(-17.72)                           # eV^2
L=L                                         # Km
P21_values = [oscillation_probability(1, 0, x, dm2, L, U) for x in E_nu]
P22_values = [oscillation_probability(1, 1, x, dm2, L, U) for x in E_nu]
P23_values = [oscillation_probability(1, 2, x, dm2, L, U) for x in E_nu]
P24_values=[]
for i in range(len(P21_values)):
  P24_value=  1-(P21_values[i]+P22_values[i]+P23_values[i])
  P24_values.append(P24_value)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(E_nu, P21_values,  color='c', alpha=0.9,label=r'$\nu_\mu \rightarrow \nu_e$')
ax.plot(E_nu, P22_values,  color='r',alpha=0.7,label=r'$\nu_\mu \rightarrow \nu_\nu$')
ax.plot(E_nu, P23_values,  color='g',alpha=0.8,label=r'$\nu_\mu \rightarrow \nu_\tau$')
ax.plot(E_nu, P24_values,  color='purple',alpha=0.6,label=r'$\nu_\mu \rightarrow \nu_S$')
ax.set_xscale('log')
ax.set_xlabel(r'Neutrino Energy $E_\nu$ (GeV)', fontsize=14)
ax.set_ylabel(r'Oscillation Probability', fontsize=14)
ax.set_title('Neutrino Oscillation Probability', fontsize=16)
ax.set_ylim([0,1])
plt.legend()
plt.show()
