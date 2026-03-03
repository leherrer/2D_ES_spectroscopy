from qutip import *
import numpy as np

from numpy import linalg as LA
from matplotlib.colors import TwoSlopeNorm
import time  # To measure execution time
import matplotlib.pyplot as plt
from matplotlib import cm
#from pylab import cm
import matplotlib as mpl

from qutip.solver.heom import HEOMSolver
from util_HAM import convert_to_xx,  lam, gamma,  kT, ham_sys, ham_sysbath , dipole0 , nsite , hbar, J
from util_RD import Hsys, mu_p0, mu_m0 , dipole, O_list, rho0, options ,Evolve_RD_all

from util_2D_eigen import R_signal_para , Fourier_Transfor,  normalize_intensities ,R_signal_para_pathway

print(rho0.shape)
gam =((1/gamma*hbar))
print(gam)



t_final, dt = 500., 10.

# Waiting time parameters
#T_init, T_final, dT = 0., 700., 700.
#Time2s = [0,100,200,500,1000,1500,2000]
Time2s = [0]


start_time = time.time()
#Rsignal_rp ,Rsignal_nr = R_signal_para(T_init, T_final, dT, t_final, dt,6)
Rsignal_rp ,Rsignal_nr = R_signal_para(Time2s, t_final, dt,40)
# Calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

np.save("Rsignal_rp_RD_J-%0.1f_l-%0.1f.npy"%(J, lam), Rsignal_rp)
np.save("Rsignal_nr_RD_J-%0.1f_l-%0.1f.npy"%(J, lam), Rsignal_nr)



omega1s, omega3s, t2s, spectra, times, Rsignal = Fourier_Transfor(Rsignal_rp,Rsignal_nr,
                                  -500., 500., 5.,
                                  -500., 500., 5.,
                                  Time2s,
                                  t_final, dt)


for t2, spectrum in zip(t2s, spectra):
                with open('2d_t2-%0.1f_RD_dt-%0.0f_tf-%0.0f_g-%0.1f_l-%0.1f_J-%0.1f.dat'%(t2,dt,t_final, gam, lam, J), 'w') as f:
                #with open('test42.dat', 'w') as f:
                    for w1 in range(len(omega1s)):
                        for w3 in range(len(omega3s)):
                            f.write('%0.8f %0.8f %0.8f\n'%(omega1s[w1], omega3s[w3], spectrum[w3,w1]))
                        f.write('\n')




