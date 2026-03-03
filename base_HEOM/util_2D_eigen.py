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
from util_HEOM import Hsys ,rho0, options  , mu_p0, mu_m0
from util_HEOM import L_matrix, N_ADO , nsite , direct_sum , pad_vector_with_zeros
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


def Resp_para_HEOM(i, j, k,t1,t2,t3):
  #Print the progress of the loop
  #print(f"Processing i={i}, j={j}, k={k}")
  #rp = mu_minus_vec @ sp.expm(L_matrix *t3/hbar )@ mu_plus_s @  sp.expm(L_matrix *t2/hbar )  @ mu_plus_s @  sp.expm(L_matrix *t1/hbar )  @mu_minus_s @ vec_rho0
  rp = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s @ vec_rho0
  nr = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_minus_s @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_plus_s @ vec_rho0
  return i, j, k, rp, nr


def Resp_para_pathway(i, j, k,t1,t2,t3):
  #Print the progress of the loop
  #print(f"Processing i={i}, j={j}, k={k}")
   ## Rephasing 
  #phi1  = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_right @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s_right @ vec_rho0
  phi1= Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_right @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s_right @ vec_rho0
  phi2= Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s_right @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s_right @ vec_rho0
  phi3= Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s_right @ vec_rho0

   ##Non-rephasing 
  phi4 = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_right @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_minus_s_right @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_plus_s_left @ vec_rho0
  phi5 = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_minus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_plus_s_left @ vec_rho0
  phi6 = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_minus_s_right @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_plus_s_left @ vec_rho0
    
  rp = phi1 + phi2 - phi3
  nr = phi4 + phi5 - phi6
  return i, j, k, rp, nr , phi1 ,phi2 ,phi3 , phi4 ,phi5, phi6



# def rho_pulse(t1,t2):
#     vec =  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s_right @ vec_rho0
#     vec =  vec.reshape((N_ADO, Nsystem,Nsystem), order="C")
#     return vec[0,:,:]

def rho_pulse(t1,t):
    if t >= t1:
        vec =  U @sp.expm(diagonal_matrix *(t-t1)/hbar ) @ U_1  @ Ado_mu_plus_s_left @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @ Ado_mu_minus_s_right @ vec_rho0
    if t< t1:
        vec = U @sp.expm(diagonal_matrix *t/hbar ) @ U_1 @ Ado_mu_minus_s_right @ vec_rho0
    aux = vec[0:Nsystem**2]
    vec =  aux.reshape((Nsystem,Nsystem), order="F")
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
                  futures.append(executor.submit(Resp_para_HEOM, i, j, k, t1, t2, t3))

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
Nsystem = nsite
print(Nsystem,nsite)    
muplus = Qobj(mu_p0)
muminus = Qobj(mu_m0)

###Dipole superoprator and vectos
mu_plus_s = (spre(muplus )  - spost(muplus)).full(order="C")
mu_minus_s = (spre(muminus )  - spost(muminus)).full(order="C")

mu_plus_s_left = (spre(muplus )).full(order="C")  ### dipole+ supeoprator accting on left
mu_plus_s_right = (spost(muplus)).full(order="C")  ### dipole+ supeoprator accting on right 
mu_minus_s_left = (spre(muminus)).full(order="C") ### dipole- supeoprator accting on left
mu_minus_s_right = (spost(muminus)).full(order="C") ### dipole- supeoprator accting on right


vec_rho0 = (rho0).reshape(N_ADO* Nsystem**2, order="C")
mu_plus_vec = (muplus.full(order="C")).reshape(Nsystem**2, order="C")
mu_minus_vec = (muminus.full(order="C")).reshape(Nsystem**2, order="C")

#vec_rho0_left = (rho0.dag().full(order="F")).reshape(Nsystem**2, order="F")
mu_plus_vec_left = (muplus.dag().full(order="C")).reshape(Nsystem**2, order="C")
mu_minus_vec_left = (muminus.dag().full(order="C")).reshape(Nsystem**2, order="C")


### Ado dipole superator and vectors 

Ado_mu_plus_s = direct_sum(mu_plus_s,N_ADO)
Ado_mu_minus_s = direct_sum(mu_minus_s,N_ADO)

Ado_mu_plus_s_left = direct_sum(mu_plus_s_left,N_ADO)
Ado_mu_plus_s_right = direct_sum(mu_plus_s_right,N_ADO)
Ado_mu_minus_s_left = direct_sum(mu_minus_s_left,N_ADO)
Ado_mu_minus_s_right = direct_sum(mu_minus_s_right,N_ADO)

Ado_mu_plus_vec = np.tile(mu_plus_vec, N_ADO)
Ado_mu_minus_vec = np.tile(mu_minus_vec, N_ADO)

Ado_mu_plus_vec_left = np.tile(mu_plus_vec_left, N_ADO)
Ado_mu_minus_vec_left = np.tile(mu_minus_vec_left, N_ADO)

## Expected values dipole 
Ado_mu_plus_vec_exp = pad_vector_with_zeros(mu_plus_vec, N_ADO)
Ado_mu_minus_vec_exp = pad_vector_with_zeros(mu_minus_vec, N_ADO)

Ado_mu_plus_vec_left_exp = pad_vector_with_zeros(mu_plus_vec_left, N_ADO)
Ado_mu_minus_vec_left_exp = pad_vector_with_zeros(mu_minus_vec_left, N_ADO)

#######

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



##########################################

index = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 57, 58, 59, 60, 66, 67, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 533, 534, 535, 536, 537, 538, 539, 542, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559]
#index = range(560)



UC = U[:,index]
#UC_1 = LA.inv(UC.T @ UC) @ UC.T
UC_1 = LA.pinv(UC)
diagonal_matrix_C = np.diag(eigenvalues[index])


def Resp_para_HEOM2(i, j, k,t1,t2,t3, U , U_1, diagonal_matrix):
  #Print the progress of the loop
  #print(f"Processing i={i}, j={j}, k={k}")
  #rp = mu_minus_vec @ sp.expm(L_matrix *t3/hbar )@ mu_plus_s @  sp.expm(L_matrix *t2/hbar )  @ mu_plus_s @  sp.expm(L_matrix *t1/hbar )  @mu_minus_s @ vec_rho0
  rp = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_plus_s @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_minus_s @ vec_rho0
  nr = Ado_mu_plus_vec_left_exp @ U @ sp.expm(diagonal_matrix *t3/hbar ) @U_1 @ Ado_mu_plus_s @  U @sp.expm(diagonal_matrix *t2/hbar ) @ U_1  @ Ado_mu_minus_s @  U @sp.expm(diagonal_matrix *t1/hbar ) @ U_1 @Ado_mu_plus_s @ vec_rho0
  return i, j, k, rp, nr


def R_signal_para2(time2_min, time2_max, dt2,
                  time_final, dt, Ncores):


  time2_max += 1e-8 # try to include time2_max
  time2s = np.arange(time2_min, time2_max, dt2)
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
                  futures.append(executor.submit(Resp_para_HEOM2, i, j, k, t1, t2, t3, UC , UC_1, diagonal_matrix_C))

      # Collect results
      for future in futures:
          i, j, k, rp, nr = future.result()
          Rsignal_rp[i, j, k] = rp
          Rsignal_nr[i, j, k] = nr


  return Rsignal_rp ,Rsignal_nr 