from qutip import *
import numpy as np
from numpy import linalg as LA
import scipy.linalg as sp
from qutip.solver.heom import HEOMSolver
from qutip.solver.heom import DrudeLorentzBath
from qutip.solver.heom import BosonicBath
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
#from util_HEOM import  Hsys, rho0, options , baths , terminator, N_ADO , mu_p0, mu_m0, NC, Nk , evolve_HEOM
from util_RD import Hsys ,rho0, options  , mu_p0, mu_m0
from util_RD import RD , nsite
import time  # To measure execution time

kB = 0.69352    # in cm-1 / K
hbar = 5308.8   # in cm-1 * fs


def Resp_para(i, j, k,t1,t2,t3):
  #Print the progress of the loop
  #print(f"Processing i={i}, j={j}, k={k}")
  #rp = mu_minus_vec @ sp.expm(L_matrix *t3/hbar )@ mu_plus_s @  sp.expm(L_matrix *t2/hbar )  @ mu_plus_s @  sp.expm(L_matrix *t1/hbar )  @mu_minus_s @ vec_rho0
  rp = mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ mu_plus_s @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @mu_minus_s @ vec_rho0
  nr = mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ mu_minus_s @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @mu_plus_s @ vec_rho0
  return i, j, k, rp, nr


def Resp_para_pathway(i, j, k,t1,t2,t3):
  #Print the progress of the loop
  #print(f"Processing i={i}, j={j}, k={k}")
   ## Rephasing 
  #phi1  = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_right @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s_right @ vec_rho0
  phi1= mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s_right @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_minus_s_right @ vec_rho0
  phi2= mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ mu_plus_s_right @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_minus_s_right @ vec_rho0
  phi3= mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @  mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_minus_s_right @ vec_rho0

   ##Non-rephasing 
  phi4 = mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s_right @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ mu_minus_s_right @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_plus_s_left @ vec_rho0
  phi5 = mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ mu_minus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_plus_s_left @ vec_rho0
  phi6 = mu_plus_vec_left @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @  mu_minus_s_right @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_plus_s_left @ vec_rho0
    
  rp = phi1 + phi2 - phi3
  nr = phi4 + phi5 - phi6
  return i, j, k, rp, nr , phi1 ,phi2 ,phi3 , phi4 ,phi5, phi6

def rho_pulse1(t1):
    vec = U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_minus_s_right @ vec_rho0
    vec =  vec.reshape((Nsystem,Nsystem), order="F")
    return vec

def rho_pulse2(t1,t2):
    vec =  U @sp.expm(diagonal_matrix *t2/hbar) @ U_1  @ mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ mu_minus_s_right @ vec_rho0
    vec =  vec.reshape((Nsystem,Nsystem), order="F")
    return vec

    


#def R_signal_para(time2_min, time2_max, dt2,
#                  time_final, dt, Ncores):
def R_signal_para(time2s,
                    time_final, dt, Ncores):

  #time2_max += 1e-8 # try to include time2_max
  #time2s = np.arange(time2_min, time2_max, dt2)
  times = np.arange(0.0, time_final, dt)

  print("--- Spectrum will require O(%d) propagations."%(len(times)*len(time2s)))

  print("--- Calculating third-order response function ..."),

  Rsignal_rp = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Rsignal_nr = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)

  print(len(times)*len(time2s)*len(times))

  #multiprocessing.set_start_method('spawn', force=True)

  with ProcessPoolExecutor(max_workers=Ncores) as executor:
      futures = []
      for i, t3 in enumerate(times):
          for j, t2 in enumerate(time2s):
              for k, t1 in enumerate(times):
                  futures.append(executor.submit(Resp_para, i, j, k, t1, t2, t3))

      # Collect results
      for future in futures:
          i, j, k, rp, nr = future.result()
          Rsignal_rp[i, j, k] = rp
          Rsignal_nr[i, j, k] = nr


  return Rsignal_rp ,Rsignal_nr ,


def R_signal_para_pathway(time2s,
                    time_final, dt, Ncores):

  #time2_max += 1e-8 # try to include time2_max
  #time2s = np.arange(time2_min, time2_max, dt2)
  times = np.arange(0.0, time_final, dt)

  print("--- Spectrum will require O(%d) propagations."%(len(times)*len(time2s)))

  print("--- Calculating third-order response function ..."),

  Rsignal_rp = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Rsignal_nr = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Phi1 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Phi2 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Phi3 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Phi4 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Phi5 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)
  Phi6 = np.zeros( (len(times),len(time2s),len(times)), dtype=complex)

  print(len(times)*len(time2s)*len(times))

  #multiprocessing.set_start_method('spawn', force=True)

  with ProcessPoolExecutor(max_workers=Ncores) as executor:
      futures = []
      for i, t3 in enumerate(times):
          for j, t2 in enumerate(time2s):
              for k, t1 in enumerate(times):
                  futures.append(executor.submit(Resp_para_pathway, i, j, k, t1, t2, t3))

      # Collect results
      for future in futures:
          i, j, k, rp, nr , phi1 ,phi2 ,phi3 , phi4 ,phi5, phi6 = future.result()
          Rsignal_rp[i, j, k] = rp
          Rsignal_nr[i, j, k] = nr
          Phi1[i, j, k] = phi1
          Phi2[i, j, k] = phi2
          Phi3[i, j, k] = phi3
          Phi4[i, j, k] = phi4
          Phi5[i, j, k] = phi5
          Phi6[i, j, k] = phi6



  return Rsignal_rp ,Rsignal_nr , Phi1, Phi2, Phi3, Phi4, Phi5, Phi6



def Fourier_Transfor(Rrp, Rnr ,
                    e1_min, e1_max, de1,
                    e3_min, e3_max, de3,
                    time2s,
                    time_final, dt):

    energy1s = np.arange(e1_min, e1_max, de1)
    energy3s = np.arange(e3_min, e3_max, de3)
    omega1s = energy1s/1
    omega3s = energy3s/1

    #time2_max += 1e-8 # try to include time2_max
    #time2s = np.arange(time2_min, time2_max, dt2)
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

def normalize_intensities(intensities):
    # Define the min and max of the array
    min_val = intensities.min()
    max_val = intensities.max()
    
    # Create an output array with the same shape
    normalized_intensities = np.zeros_like(intensities, dtype=float)
    
    # Normalize positive values
    positive_mask = intensities > 0
    normalized_intensities[positive_mask] = intensities[positive_mask] / max_val
    
    # Normalize negative values
    negative_mask = intensities < 0
    normalized_intensities[negative_mask] = intensities[negative_mask] / -min_val
    
    return normalized_intensities

### Liouvulle
L_matrix = RD.full(order="C")
Nsystem = nsite
muplus = Qobj(mu_p0)
muminus = Qobj(mu_m0)

###Dipole superoprator and vectos
mu_plus_s = (spre(muplus )  - spost(muplus)).full(order="C")
mu_minus_s = (spre(muminus )  - spost(muminus)).full(order="C")

mu_plus_s_left = (spre(muplus )).full(order="C")  ### dipole+ supeoprator accting on left
mu_plus_s_right = (spost(muplus)).full(order="C")  ### dipole+ supeoprator accting on right 
mu_minus_s_left = (spre(muminus)).full(order="C") ### dipole- supeoprator accting on left
mu_minus_s_right = (spost(muminus)).full(order="C") ### dipole- supeoprator accting on right

vec_rho0 = (rho0.full(order="C")).reshape(Nsystem**2, order="C")
mu_plus_vec = (muplus.full(order="C")).reshape(Nsystem**2, order="C")
mu_minus_vec = (muminus.full(order="C")).reshape(Nsystem**2, order="C")

vec_rho0_left = (rho0.dag().full(order="C")).reshape(Nsystem**2, order="C")
mu_plus_vec_left = (muplus.dag().full(order="C")).reshape(Nsystem**2, order="C")
mu_minus_vec_left = (muminus.dag().full(order="C")).reshape(Nsystem**2, order="C")

## Eigenvalues
start_time = time.time()
print("Calculatin eigenvalues")
eigenvalues, eigenvectors = LA.eig(L_matrix)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")

diagonal_matrix = np.diag(eigenvalues)


U = eigenvectors
U_1 = LA.inv(eigenvectors)
