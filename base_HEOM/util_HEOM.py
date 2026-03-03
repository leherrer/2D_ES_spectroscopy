from qutip import *
import numpy as np
from numpy import linalg as LA
from qutip.solver.heom import HEOMSolver
from qutip.solver.heom import DrudeLorentzBath
from qutip.solver.heom import BosonicBath
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from util_HAM import convert_to_xx, lam, gamma,  kT, ham_sys, ham_sysbath , dipole0 , nsite , nbath , direct_sum ,pad_vector_with_zeros


kB = 0.69352    # in cm-1 / K
hbar = 5308.8   # in cm-1 * fs

def cot(x):
    return 1./np.tan(x)

def Bath_HEOM(lam,gamma,T, O_list,Nk):
    Ndim = 4
    #Nk = 2
    pref = 1.
    b = 1/T
    ckAR = [ lam * gamma * (cot(gamma / (2 * T)))]
    ckAR.extend([(4 * lam * gamma * T *  2 * np.pi * k * T / (( 2 * np.pi * k * T)**2 - gamma**2)) for k in range(1,Nk+1)])

    vkAR = [gamma]
    vkAR.extend([2 * np.pi * k * T for k in range(1,Nk+1)])

    ckAI = [lam * gamma * (-1.0)]

    vkAI = [gamma]



    NR = len(ckAR)
    NI = len(ckAI)
    QQ = []
    ckAR2 = []
    ckAI2 = []
    vkAR2 = []
    vkAI2 = []
    #for m in range(1,Ndim-1):
    for m in range(len(O_list)):
        #QQ.extend([ basis(Ndim,m)*basis(Ndim,m).dag() for kk in range(NR)])
        QQ.extend([ Qobj(O_list[m]) for kk in range(NR)])
        ckAR2.extend(ckAR)    
        vkAR2.extend(vkAR)

    #for m in range(1,Ndim-1):
    for m in range(len(O_list)):
        #QQ.extend([ basis(Ndim,m)*basis(Ndim,m).dag() for kk in range(NI)])
        QQ.extend([ Qobj(O_list[m]) for kk in range(NI)])
        ckAI2.extend(ckAI)
        vkAI2.extend(vkAI)

    #Q_list=[basis(Ndim,m)*basis(Ndim,m).dag() for m in range(1,Ndim-1)]
    Q_list=[ Qobj(O_list[m])  for m in range(len(O_list))]
    L_bnd = 0.0*spre(Q_list[0])*spost(Q_list[0].dag())
    for Q1 in Q_list:
        op = -2*spre(Q1)*spost(Q1.dag()) + spre(Q1.dag()*Q1) + spost(Q1.dag()*Q1)

        approx_factr = ((2 * lam / (b * gamma)) - 1j*lam) 

        approx_factr -=  lam * gamma * (-1.0j + cot(gamma / (2 * T)))/gamma
        for k in range(1,Nk+1):
            vk = 2 * np.pi * k * T

            approx_factr -= ((pref * 4 * lam * gamma * T * vk / (vk**2 - gamma**2))/ vk)

        L_bnd += -approx_factr*op
        
    return L_bnd , QQ, ckAR2, ckAI2, vkAR2, vkAI2
        
def Baths(O_list, lam,gamma, kT, Nk):
    baths = []
    terminator = []
    for i in range(len(O_list)):
        bath_aux = DrudeLorentzBath(O_list[i], lam, gamma, kT, Nk)
        baths.append( bath_aux )
        terminator.append( bath_aux.terminator()[1])
    
    return baths, terminator
    
    
def evolve_HEOM(rho0, t0 , tf, dt, Hsys ,N_ADO , baths , terminator, NC , options):
    
    HL = liouvillian(Hsys)
    for i in range(len(baths)):
        HL += terminator[i] 
    
    ##Transpose the ADOS 
    for i in range(N_ADO):
        rho0[i,:,:] = np.transpose(rho0[i,:,:])
        
        
    solver = HEOMSolver(HL, baths, max_depth=NC, options=options)    
    tlist = np.arange(t0,tf+dt, dt) /5308.0  #fs  
    result = solver.run(rho0, tlist)
    
    ado_state = result.ado_states[-1]
    
    ados = [
        ado_state.extract(label).full()
        for label in ado_state.filter() ]
    
    
    return np.array(ados)





def N_ADOS(Hsys,baths,NC):
    HL = liouvillian(Hsys)
    Nsystem = Hsys.shape[0]
    max_depth = NC  # maximum hierarchy depth to retain
    options = {"nsteps": 15_000, "store_ados": True}
    solver = HEOMSolver(HL, baths, max_depth=max_depth, options=options)

    Liovvile = solver._calculate_rhs()
    L_matrix = Liovvile.__call__(0)
    L_matrix = L_matrix.full()
    NN = len(L_matrix)
    
    N_ADO = NN // (Nsystem**2)
    return N_ADO
    
    
def strike_evolve(rho_matrix,t,direction, pm, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options ):
    dt = 5.0
  
  #print(N,NN,k)
    if direction == "left" and pm =="+":
        rho_timex_mu =  [np.matmul(mu_p, rho_matrix[i,:,:] ) for i in range(N_ADO)] 
    if direction == "right" and pm =="+":
        rho_timex_mu =  [np.matmul(rho_matrix[i,:,:], mu_p ) for i in range(N_ADO)] 
    if direction == "left" and pm =="-":
        rho_timex_mu =  [np.matmul(mu_m, rho_matrix[i,:,:] ) for i in range(N_ADO)] 
    if direction == "right" and pm =="-":
        rho_timex_mu =  [np.matmul(rho_matrix[i,:,:], mu_m ) for i in range(N_ADO)] 
    #print("input", rho_timex_mu)
    rho_evolved = evolve_HEOM(np.array(rho_timex_mu), 0 , t, dt, Hsys, N_ADO , baths , terminator, NC , options)   

    return rho_evolved    




def response_function(t1,t2,t3, pathway, rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options):
    Nsystem = Hsys.shape[0]

    if pathway == "1":
        rho = strike_evolve(     rho0,t1,"right", "-",  mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t2 ,"left" , "+",  mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t3,"right", "+",   mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "2":
        rho = strike_evolve(     rho0,t1,"right", "-", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t2,"right" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t3,"left" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "3":
        rho = strike_evolve(     rho0,t1,"right", "-", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t2,"left" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t3,"left" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "4":
        rho = strike_evolve(     rho0,t1,"left", "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t2,"right" , "-", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t3,"right" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "5":
        rho = strike_evolve(     rho0,t1,"left", "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t2,"left" , "-",mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t3,"left" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "6":
        rho = strike_evolve(    rho0,t1,"left", "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t2,"right" , "-", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho = strike_evolve(     rho,t3,"left" , "+", mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options )
        rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value


def R_signal(time2_min, time2_max, dt2,
                    time_final, dt, rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options):


    time2_max += 1e-8 # try to include time2_max
    time2s = np.arange(time2_min, time2_max, dt2)
    times = np.arange(0.0, time_final, dt)

    print("--- Spectrum will require O(%d) propagations."%(len(times)*len(time2s)))

    print("--- Calculating third-order response function ..."),

    Rsignal_rp = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
    Rsignal_nr = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)

    phi1 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
    phi2 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
    phi3 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
    phi4 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
    phi5 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
    phi6 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)


    print(len(times)*len(time2s)*len(times))

    for i,t3 in enumerate(times):
        for j,t2 in enumerate(time2s):
            for k,t1 in enumerate(times):
                print(i,j,k)
                phi1[i,j,k] = response_function(t1,t2,t3,"1", rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options)
                phi2[i,j,k] = response_function(t1,t2,t3,"2", rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options)
                phi3[i,j,k] = response_function(t1,t2,t3,"3", rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options)
                phi4[i,j,k] = response_function(t1,t2,t3,"4", rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options)
                phi5[i,j,k] = response_function(t1,t2,t3,"5", rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options)
                phi6[i,j,k] = response_function(t1,t2,t3,"6", rho0, mu_p, mu_m, N_ADO, Hsys , baths , terminator, NC , options)
                Rsignal_rp[i,j,k] = phi1[i,j,k] + phi2[i,j,k] - phi3[i,j,k]
                Rsignal_nr[i,j,k] = phi4[i,j,k] + phi5[i,j,k] - phi6[i,j,k]

        #Rsignal_rp[i,j,k] = #response_function(t1,t2,t3,"1") # + response_function(t1,t2,t3,"2") + np.conj( response_function(t1,t2,t3,"4"))
        #Rsignal_nr[i,j,k] = 0#response_function(t1,t2,t3,"4") + response_function(t1,t2,t3,"5") + np.conj( response_function(t1,t2,t3,"1"))


    return Rsignal_rp ,Rsignal_nr , phi1 , phi2, phi3, phi4 ,phi5,phi6    
    

def Fourier_Transfor(Rrp, Rnr ,
                    e1_min, e1_max, de1,
                    e3_min, e3_max, de3,
                    time2_min, time2_max, dt2,
                    time_final, dt):

    energy1s = np.arange(e1_min, e1_max, de1)
    energy3s = np.arange(e3_min, e3_max, de3)
    omega1s = energy1s/1
    omega3s = energy3s/1

    time2_max += 1e-8 # try to include time2_max
    time2s = np.arange(time2_min, time2_max, dt2)
    times = np.arange(0.0, time_final, dt)

    spectrum = np.zeros( (len(omega3s),len(time2s),len(omega1s)) )

    Rsignal = []
    Rsignal.append(Rrp)
    Rsignal.append(Rnr)

    expi1 = np.exp(1j*(1/hbar)*np.outer(omega1s,times))
    expi1[:,0] *= 0.5*dt
    expi1[:,1:] *= dt
    expi3 = np.exp(1j*(1/hbar)*np.outer(omega3s,times))
    expi3[:,0] *= 0.5*dt
    expi3[:,1:] *= dt
    #np.save( "Rsignal_rnR.npy", Rsignal[1][:,:,:])
    #np.save( "Rsignal_rpR.npy", Rsignal[0][:,:,:])
    spectrum =  np.einsum('ws,xu,uts->xtw',expi1,expi3,Rnr).real
    spectrum += np.einsum('ws,xu,uts->xtw',expi1.conj(),expi3,Rrp).real
    #np.save( "spectrumR.npy" , spectrum[:,1,:] )


    print("done.")

    spectra = []
    for t2 in range(len(time2s)):
        spectra.append( spectrum[:,t2,:] )

    return energy1s, energy3s, time2s, spectra, times, Rsignal    
    


Hsys = Qobj(ham_sys)
#Q1, Q2 = Qobj(ham_sysbath[0]) , Qobj(ham_sysbath[1])
mu_p0 = np.tril(dipole0,0)
mu_m0 = np.triu(dipole0,0)
dipole = Qobj(dipole0)

O_list = [Qobj(ham_sysbath[i]) for i in range(nbath)]

NC = 3 # cut off parameter for the bath
Nk = 1

#L_bnd , QQ, ckAR2, ckAI2, vkAR2, vkAI2 = Bath_HEOM(lam,gamma,kT, ham_sysbath,Nk)
baths, terminator = Baths(O_list, lam,gamma, kT, Nk)
N_ADO = N_ADOS(Hsys,baths,NC)

rho0 = basis(nsite,0) * basis(nsite,0).dag()
options = {"nsteps": 15_000, "store_ados": True, "progress_bar": False}
rho = [rho0.full()]
for i in range(1,N_ADO):
    rho.append(0*rho0.full())
rho0 = np.array(rho) 



### Construc superoprator 

HL = liouvillian(Hsys)
for i in range(len(baths)):
    HL += terminator[i] 

##Transpose the ADOS 
#for i in range(N_ADO):
#    rho0[i,:,:] = np.transpose(rho0[i,:,:])
        
solver = HEOMSolver(HL, baths, max_depth=NC, options=options)    

Liovvile = solver._calculate_rhs()
L_matrix = Liovvile.__call__(0)
L_matrix = L_matrix.full(order="C")
#N_ADO = len(L_matrix)


#print(rho0)
#print(N_ADO)     
    
