import numpy as np
from numpy import linalg as LA
import scipy.linalg as sp
from qutip import *

def convert_to_xx(ham_sys_x, ham_sysbath_x, dipole_x):
    import scipy
    # Two-exciton Hamiltonian
    nx = ham_sys_x.shape[0]
    nxx = int(nx*(nx-1)/2)
    ham_sys_xx = np.zeros((nxx,nxx))
    mn = 0
    for m in range(nx):
        for n in range(m):
            if m != n:
                ham_sys_xx[mn,mn] = ham_sys_x[m,m] + ham_sys_x[n,n]
                op = 0
                for o in range(nx):
                    for p in range(o):
                        if o != p:
                            ham_sys_xx[mn,op] = ham_sys_x[m,o]*(n==p) + ham_sys_x[n,p]*(m==o)
                        op += 1
            mn += 1

    ham_sys = scipy.linalg.block_diag([[0]], ham_sys_x, ham_sys_xx)
    nsite = ham_sys.shape[0]

    nbath = len(ham_sysbath_x)
    ham_sysbath = []
    for b in range(nbath):
        ham_sysbath_xx_b = np.zeros((nxx,nxx))
        mn = 0
        for m in range(nx):
            for n in range(m):
                ham_sysbath_xx_b[mn,mn] = ham_sysbath_x[b][m,m] + ham_sysbath_x[b][n,n]
                mn += 1
        ham_sysbath.append(
            scipy.linalg.block_diag([[0]], ham_sysbath_x[b], ham_sysbath_xx_b) )

    dipole_xx = np.zeros((nx,nxx))
    for i in range(nx):
        mn = 0
        for m in range(nx):
            for n in range(m):
                dipole_xx[i,mn] = dipole_x[m]*(i==n) + dipole_x[n]*(i==m)
                mn += 1

    dipole = np.zeros((1+nx+nxx,1+nx+nxx))
    for i in range(nx):
        dipole[0,i+1] = dipole[i+1,0] = dipole_x[i]
        mn = 0
        for m in range(nx):
            for n in range(m):
                dipole[i+1,mn+1+nx] = dipole[mn+1+nx,i+1] = dipole_xx[i,mn]

    return ham_sys, ham_sysbath, dipole


def hamiltonian_custom_order(N, energies, couplings):
    # Define the annihilation operator B and the creation operator B_dagger
    B = basis(2, 0) * basis(2, 1).dag()
    B_dagger = B.dag()
    
    # Initialize Hamiltonian
    H = 0
    
    # On-site energy terms: sum(e_i * B_i^dagger * B_i)
    for i in range(N):
        operators = [qeye(2) for _ in range(N)]
        operators[i] = B_dagger * B  # Place B_i^dagger * B_i at position i
        H += energies[i] * tensor(*operators)
    
    # Coupling terms: sum(J_ij * B_i^dagger * B_j) for i != j
    for i in range(N):
        for j in range(N):
            if i != j:
                operators_i = [qeye(2) for _ in range(N)]
                operators_j = [qeye(2) for _ in range(N)]
                
                operators_i[i] = B_dagger  # B_i^dagger at position i
                operators_j[j] = B         # B_j at position j
                
                H += couplings[i][j] * tensor(*operators_i) * tensor(*operators_j)
    
    # Define custom order of states up to the second excitation manifold
    states_ordered = ['0' * N]  # Start with the no-excitation state (|000...0⟩)
    
    # Add single-excitation states (e.g., |100⟩, |010⟩, |001⟩ for N=3)
    states_ordered += [f"{'0' * i}1{'0' * (N - i - 1)}" for i in range(N)]
    
    # Add double-excitation states (e.g., |110⟩, |101⟩, |011⟩ for N=3)
    for i in range(N):
        for j in range(i + 1, N):
            state = ['0'] * N
            state[i] = '1'
            state[j] = '1'
            states_ordered.append(''.join(state))
    
    # Generate the reordering index based on the custom state order
    all_states = [f"{format(i, f'0{N}b')}" for i in range(2**N)]
    index_map = [all_states.index(state) for state in states_ordered]
    
    # Reorder and truncate the Hamiltonian using the index map
    #H_truncated = Qobj(np.array(H.full())[index_map][:, index_map])
    H_truncated = np.array(H.full())[index_map][:, index_map]

    # Create ordered labels for the truncated basis states
    labels_ordered = [f"|{state}⟩" for state in states_ordered]
    
    return H_truncated, labels_ordered


def mu_operator_ordered(N, mu_vec):
    # The creation-annihilation operator
    op = basis(2, 0) * basis(2, 1).dag() + basis(2, 1) * basis(2, 0).dag()
    
    # Initialize the sum operator
    mu = 0
    
    # Iterate over each qubit position to build the operator
    for i in range(N):
        operators = [qeye(2) for _ in range(N)]
        operators[i] = mu_vec[i]*op
        mu += tensor(*operators)
    
    # Define custom order of states up to the second excitation manifold
    states_ordered = ['0' * N]  # Start with the no-excitation state (|000...0⟩)
    
    # Add single-excitation states (e.g., |100⟩, |010⟩, |001⟩ for N=3)
    states_ordered += [f"{'0' * i}1{'0' * (N - i - 1)}" for i in range(N)]
    
    # Add double-excitation states (e.g., |110⟩, |101⟩, |011⟩ for N=3)
    for i in range(N):
        for j in range(i + 1, N):
            state = ['0'] * N
            state[i] = '1'
            state[j] = '1'
            states_ordered.append(''.join(state))
    
    # Generate the reordering index based on the custom state order
    all_states = [f"{format(i, f'0{N}b')}" for i in range(2**N)]
    index_map = [all_states.index(state) for state in states_ordered]

    # Reorder and truncate the Hamiltonian using the index map
    #H_truncated = Qobj(np.array(H.full())[index_map][:, index_map])
    mu_reordered = np.array(mu.full())[index_map][:, index_map]

    # Create ordered labels for the truncated basis states
    labels_ordered = [f"|{state}⟩" for state in states_ordered]
    return mu_reordered, labels_ordered


def sys_bath_ordered(N, site):
    # The creation-annihilation operator
    op = basis(2, 1) * basis(2, 1).dag()
    operators = [qeye(2) for _ in range(N)]
    operators[site] = op
    # Initialize the sum operator
    Q = tensor(*operators)
    
    # Iterate over each qubit position to build the operator
    # Define custom order of states up to the second excitation manifold
    states_ordered = ['0' * N]  # Start with the no-excitation state (|000...0⟩)
    
    # Add single-excitation states (e.g., |100⟩, |010⟩, |001⟩ for N=3)
    states_ordered += [f"{'0' * i}1{'0' * (N - i - 1)}" for i in range(N)]
    
    # Add double-excitation states (e.g., |110⟩, |101⟩, |011⟩ for N=3)
    for i in range(N):
        for j in range(i + 1, N):
            state = ['0'] * N
            state[i] = '1'
            state[j] = '1'
            states_ordered.append(''.join(state))
    
    # Generate the reordering index based on the custom state order
    all_states = [f"{format(i, f'0{N}b')}" for i in range(2**N)]
    index_map = [all_states.index(state) for state in states_ordered]

    # Reorder and truncate the Hamiltonian using the index map
    #H_truncated = Qobj(np.array(H.full())[index_map][:, index_map])
    Q_reordered = np.array(Q.full())[index_map][:, index_map]

    # Create ordered labels for the truncated basis states
    labels_ordered = [f"|{state}⟩" for state in states_ordered]
    return Q_reordered, labels_ordered


def sys_bath_list(N, site_list):
    list_Q = []
    for site in site_list:
        list_Q.append(sys_bath_ordered(N,site-1)[0])
    labels = sys_bath_ordered(N,0)[1]
    return list_Q ,labels



kB = 0.69352    # in cm-1 / K
hbar = 5308.8   # in cm-1 * fs
J = -200

# ham_sys_x =  np.array([[-50, -100,0],
#                        [-100, 50,0],
#                         [0,0,0]])

ham_sys_x =  np.array([[-50, J],
                        [J, 50]])


#ham_sys_x =  np.array([[505, -94.8, 5.5],
#                      [-94.8, 425 ,29.8],
#                      [ 5.5,   29.8 ,195]])


# ham_sys_x =  np.array([[410, -87.7, 5.5],
#                        [-87.7, 530 ,30.8],
#                        [ 5.5,  30.8 ,210]])

# ham_sys_x =  np.array([[0.0, 14.845, 2.411, 0.024, 0.032, 0.049, 0.205, -0.047, 0.004, 0.088],
# [14.845, 294.392, 11.297, -2.198, -1.037, -0.140, -0.352, 0.225, 0.140, -0.122],
# [2.411, 11.297, 299.232, 7.032, 1.195, 0.197, 0.553, -0.212, -0.097, 0.153],
# [0.024, -2.198, 7.032, 306.491, -19.968, -2.949, -1.295, 0.790, 0.362, -0.571],
# [0.032, -1.037, 1.195, -19.968, 462.156, -3.438, -6.226, 1.957, 0.861, -0.394],
# [0.049, -0.140, 0.197, -2.949, -3.438, 290.360, -12.254, -0.252, 0.059, 0.886],
# [0.205, -0.352, 0.553, -1.295, -6.226, -12.254, -656.535, 65.722, 9.242, -1.358],
# [-0.047, 0.225, -0.212, 0.790, 1.957, -0.252, 65.722, 878.017, -86.544, 2.114],
# [0.004, 0.140, -0.097, 0.362, 0.861, 0.059, 9.242, -86.544, 647.536, 1.761],
# [0.088, -0.122, 0.153, -0.571, -0.394, 0.886, -1.358, 2.114, 1.761, -780.744]])

#dipole_x = np.array([0.1,0.1,0.1])

#dipole_x = np.array([1.0,0.391,-0.312])
dipole_x = np.array([1.0,-0.2])

#dipole_x = np.array([ 1,1,1,1,1,1,1,1,1,1])
coupling_sites = [1,2]
nx = ham_sys_x.shape[0]
nbath = len(coupling_sites)

Energies = ham_sys_x.diagonal()
Couplings = ham_sys_x
Dipole = dipole_x

### Double Excitation Hamiltonian construction 
ham_sys , labels= hamiltonian_custom_order(nx, Energies, Couplings)
dipole0, labels_dip = mu_operator_ordered(nx, Dipole)
ham_sysbath, labels_sysbath = sys_bath_list(nx, coupling_sites)

### Check labels are correct  ###
print("Label correct", labels ==labels_dip ==labels_sysbath)

nsite = ham_sys.shape[0]

lam = 100.
gamma = hbar* 1./100. # in 1/fs
kT = kB*77.
b = 1/kT


