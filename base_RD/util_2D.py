from qutip import *
import numpy as np
from numpy import linalg as LA
from qutip.solver.heom import HEOMSolver
from qutip.solver.heom import DrudeLorentzBath
from qutip.solver.heom import BosonicBath
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from util_RD import Hsys ,rho0, options ,  Q1, Q2, Evolve_RD , mu_p0, mu_m0

kB = 0.69352    # in cm-1 / K
hbar = 5308.8   # in cm-1 * fs

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

def strike_evolve_RD(rho_matrix,t,direction, pm, mu_p, mu_m,  Hsys,  Q1, Q2, options ):
    dt = 1.0
  #print(N,NN,k)
    if direction == "left" and pm =="+":
        rho_timex_mu =  np.matmul(mu_p, rho_matrix.full()) 
    if direction == "right" and pm =="+":
        rho_timex_mu =  np.matmul(rho_matrix.full(), mu_p )
    if direction == "left" and pm =="-":
        rho_timex_mu =  np.matmul(mu_m, rho_matrix.full() ) 
    if direction == "right" and pm =="-":
        rho_timex_mu =  np.matmul(rho_matrix.full(), mu_m ) 
    #print("input", rho_timex_mu)
    rho_timex_mu = Qobj(rho_timex_mu)
    rho_evolved =  Evolve_RD( rho_timex_mu, t, dt, Hsys, Q1, Q2, options )
    return rho_evolved    


def response_function(t1,t2,t3, pathway, rho0, mu_p, mu_m,  Hsys,  Q1, Q2, options ):
    Nsystem = Hsys.shape[0]

    if pathway == "1":
        rho = strike_evolve_RD(     rho0,t1,"right", "-",  mu_p, mu_m,  Hsys,  Q1, Q2, options )
        rho = strike_evolve_RD(     rho,t2 ,"left" , "+",  mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t3,"right", "+",   mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        #rho_final =  rho[0,:,:].reshape(Nsystem,Nsystem)
        rho_final = rho.full()
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "2":
        rho = strike_evolve_RD(     rho0,t1,"right", "-", mu_p, mu_m, Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t2,"right" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t3,"left" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho_final = rho.full()
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "3":
        rho = strike_evolve_RD(     rho0,t1,"right", "-", mu_p, mu_m,  Hsys,  Q1, Q2, options )
        rho = strike_evolve_RD(     rho,t2,"left" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t3,"left" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho_final = rho.full()
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "4":
        rho = strike_evolve_RD(     rho0,t1,"left", "+", mu_p, mu_m,  Hsys,  Q1, Q2, options )
        rho = strike_evolve_RD(     rho,t2,"right" , "-", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t3,"right" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options )
        rho_final = rho.full()
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "5":
        rho = strike_evolve_RD(     rho0,t1,"left", "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t2,"left" , "-",mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t3,"left" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options )
        rho_final = rho.full()
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

    if pathway == "6":
        rho = strike_evolve_RD(    rho0,t1,"left", "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t2,"right" , "-", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho = strike_evolve_RD(     rho,t3,"left" , "+", mu_p, mu_m,  Hsys,  Q1, Q2, options  )
        rho_final = rho.full()
        expected_value = np.trace( np.matmul(mu_m,rho_final) )

        return expected_value

def calculate_response_para(i, j, k, t1, t2, t3):

    # Print the progress of the loop
    print(f"Processing i={i}, j={j}, k={k}")
    
    phi1 = response_function(t1, t2, t3, "1", rho0, mu_p0, mu_m0,  Hsys, Q1, Q2, options )
    phi2 = response_function(t1, t2, t3, "2", rho0, mu_p0, mu_m0,  Hsys,  Q1, Q2, options )
    phi3 = response_function(t1, t2, t3, "3", rho0, mu_p0, mu_m0,  Hsys,  Q1, Q2, options )
    phi4 = response_function(t1, t2, t3, "4", rho0, mu_p0, mu_m0,  Hsys,  Q1, Q2, options )
    phi5 = response_function(t1, t2, t3, "5", rho0, mu_p0, mu_m0,  Hsys,  Q1, Q2, options )
    phi6 = response_function(t1, t2, t3, "6", rho0, mu_p0, mu_m0,  Hsys,  Q1, Q2, options )
    return i, j, k, phi1, phi2, phi3, phi4, phi5, phi6
    
    
def R_signal_para(time2_min, time2_max, dt2,
                    time_final, dt, Ncores):


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
    
    #multiprocessing.set_start_method('spawn', force=True)
    
    with ProcessPoolExecutor(max_workers=Ncores) as executor:
        futures = []
        for i, t3 in enumerate(times):
            for j, t2 in enumerate(time2s):
                for k, t1 in enumerate(times):
                    futures.append(executor.submit(calculate_response_para, i, j, k, t1, t2, t3))

        # Collect results
        for future in futures:
            i, j, k, p1, p2, p3, p4, p5, p6 = future.result()
            phi1[i, j, k] = p1
            phi2[i, j, k] = p2
            phi3[i, j, k] = p3
            phi4[i, j, k] = p4
            phi5[i, j, k] = p5
            phi6[i, j, k] = p6
    
    

    Rsignal_rp = phi1 + phi2 - phi3
    Rsignal_nr = phi4 + phi5 - phi6

        #Rsignal_rp[i,j,k] = #response_function(t1,t2,t3,"1") # + response_function(t1,t2,t3,"2") + np.conj( response_function(t1,t2,t3,"4"))
        #Rsignal_nr[i,j,k] = 0#response_function(t1,t2,t3,"4") + response_function(t1,t2,t3,"5") + np.conj( response_function(t1,t2,t3,"1"))


    return Rsignal_rp ,Rsignal_nr , phi1 , phi2, phi3, phi4 ,phi5 ,phi6 


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
                phi1[i,j,k] = response_function(t1,t2,t3,"1", rho0, mu_p, mu_m,  Hsys, L_list_A1, L_list_A2, options )
                phi2[i,j,k] = response_function(t1,t2,t3,"2", rho0, mu_p, mu_m,  Hsys, L_list_A1, L_list_A2, options )
                phi3[i,j,k] = response_function(t1,t2,t3,"3", rho0, mu_p, mu_m,  Hsys, L_list_A1, L_list_A2, options )
                phi4[i,j,k] = response_function(t1,t2,t3,"4", rho0, mu_p, mu_m,  Hsys, L_list_A1, L_list_A2, options)
                phi5[i,j,k] = response_function(t1,t2,t3,"5", rho0, mu_p, mu_m,  Hsys, L_list_A1, L_list_A2, options)
                phi6[i,j,k] = response_function(t1,t2,t3,"6", rho0, mu_p, mu_m,  Hsys, L_list_A1, L_list_A2, options)
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


