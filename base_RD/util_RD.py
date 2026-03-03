from qutip import *
import numpy as np
from numpy import linalg as LA
from util_HAM import convert_to_xx, lam, gamma,  kT, b, ham_sys, ham_sysbath , dipole0 , nsite , nbath


     
def calculate_DL(w):
    if w == 0:
        DL = 2 * np.pi * 2.0 * lam / (np.pi * gamma * b)
    else:
        DL = 2 * np.pi * (2.0 * lam * gamma * w / (np.pi * (w**2 + gamma**2)))*((1 / (np.exp(w * b) - 1)) + 1)
    return DL
    

    
def Evolve_RD( rho0, tf, dt, Hsys, Q1, Q2, options ):
    tlist = np.arange(0,tf+dt, dt) /5308  #fs                                                                                   
    #tlist = np.linspace(0,1,10000) #ps
    #print(Hsys,rho0)
    #result =  brmesolve(Hsys, rho0, tlist, a_ops=[[Q1, calculate_DL],[Q2, calculate_DL]] , sec_cutoff=-1, options = options) ##sec_cutoff=-1
    result =  brmesolve(Hsys, rho0, tlist, a_ops=[[Q1, calculate_DL],[Q2, calculate_DL]], options = options)


    return result.states[-1]
    

def Evolve_RD_all( rho0, tf, dt, Hsys, Q1, Q2, options ):
    tlist = np.arange(0,tf+dt, dt) /5308  #fs                                                                                   
    #tlist = np.linspace(0,1,10000) #ps
    #print(Hsys,rho0)
    #result =  brmesolve(Hsys, rho0, tlist, a_ops=[[Q1, calculate_DL],[Q2, calculate_DL]] ,sec_cutoff=-1 ,options = options)
    result =  brmesolve(Hsys, rho0, tlist, a_ops=[[Q1, calculate_DL],[Q2, calculate_DL]], options = options)
    return result.states


#####
Hsys = Qobj(ham_sys)
#Q1, Q2 = Qobj(ham_sysbath[0]) , Qobj(ham_sysbath[1])
mu_p0 = np.tril(dipole0,0)
mu_m0 = np.triu(dipole0,0)
dipole = Qobj(dipole0)


### Bath operator

O_list = [Qobj(ham_sysbath[i]) for i in range(nbath)]
a_ops = [[Q, calculate_DL] for Q in O_list] 

#RD = bloch_redfield_tensor(Hsys, a_ops=[[Q1, calculate_DL],[Q2, calculate_DL]], sec_cutoff=-1 , fock_basis=True )
RD = bloch_redfield_tensor(Hsys, a_ops=a_ops, sec_cutoff=0.001 , fock_basis=True ) ## Lindland 
#RD = bloch_redfield_tensor(Hsys, a_ops=a_ops, sec_cutoff=-1 , fock_basis=True ) ##Redflied

rho0 = basis(nsite,0) * basis(nsite,0).dag()
options = {"nsteps": 15_000, "progress_bar": False}    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
