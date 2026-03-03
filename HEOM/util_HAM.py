import numpy as np
import scipy.linalg as sp
from qutip import basis, tensor, qeye


class SystemHamiltonian:
    """
    Constructs system Hamiltonian, dipole operator,
    and system-bath coupling operators in the
    truncated basis (ground + 1-exciton + 2-exciton manifold).
    """

    kB = 0.69352     # cm^-1 / K
    hbar = 5308.8    # cm^-1 * fs

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self,
                 ham_sys_x,
                 dipole_x,
                 coupling_sites,
                 lam,
                 gamma,
                 temperature):

        self.ham_sys_x = np.array(ham_sys_x)
        self.dipole_x = np.array(dipole_x)
        self.coupling_sites = coupling_sites

        self.lam = lam
        self.gamma = gamma
        self.temperature = temperature
        self.kT = self.kB * temperature

        self.nx = self.ham_sys_x.shape[0]
        self.nbath = len(coupling_sites)

        # Build everything
        self.ham_sys, self.labels = self._build_hamiltonian()
        self.dipole, self.labels_dip = self._build_dipole()
        self.ham_sysbath, self.labels_sysbath = self._build_sys_bath()

        self.nsite = self.ham_sys.shape[0]

    # ==========================================================
    # Basis construction utilities
    # ==========================================================

    def _generate_basis_order(self):
        """Generate custom basis ordering up to 2 excitations."""

        N = self.nx

        states_ordered = ['0' * N]

        # Single excitations
        states_ordered += [
            f"{'0' * i}1{'0' * (N - i - 1)}"
            for i in range(N)
        ]

        # Double excitations
        for i in range(N):
            for j in range(i + 1, N):
                state = ['0'] * N
                state[i] = '1'
                state[j] = '1'
                states_ordered.append(''.join(state))

        all_states = [format(i, f'0{N}b') for i in range(2**N)]
        index_map = [all_states.index(state) for state in states_ordered]

        labels = [f"|{state}⟩" for state in states_ordered]

        return index_map, labels

    # ==========================================================
    # Hamiltonian construction
    # ==========================================================

    def _build_hamiltonian(self):

        N = self.nx
        energies = self.ham_sys_x.diagonal()
        couplings = self.ham_sys_x

        B = basis(2, 0) * basis(2, 1).dag()
        B_dagger = B.dag()

        H = 0

        # On-site energies
        for i in range(N):
            ops = [qeye(2) for _ in range(N)]
            ops[i] = B_dagger * B
            H += energies[i] * tensor(*ops)

        # Couplings
        for i in range(N):
            for j in range(N):
                if i != j:
                    ops_i = [qeye(2) for _ in range(N)]
                    ops_j = [qeye(2) for _ in range(N)]
                    ops_i[i] = B_dagger
                    ops_j[j] = B
                    H += couplings[i][j] * tensor(*ops_i) * tensor(*ops_j)

        index_map, labels = self._generate_basis_order()

        H_truncated = np.array(H.full())[index_map][:, index_map]

        return H_truncated, labels

    # ==========================================================
    # Dipole operator
    # ==========================================================

    def _build_dipole(self):

        N = self.nx
        mu_vec = self.dipole_x

        op = basis(2, 0) * basis(2, 1).dag() + \
             basis(2, 1) * basis(2, 0).dag()

        mu = 0

        for i in range(N):
            ops = [qeye(2) for _ in range(N)]
            ops[i] = mu_vec[i] * op
            mu += tensor(*ops)

        index_map, labels = self._generate_basis_order()

        mu_reordered = np.array(mu.full())[index_map][:, index_map]

        return mu_reordered, labels

    # ==========================================================
    # System-bath coupling operators
    # ==========================================================

    def _build_sys_bath(self):

        N = self.nx
        index_map, labels = self._generate_basis_order()

        Q_list = []

        for site in self.coupling_sites:

            op = basis(2, 1) * basis(2, 1).dag()

            ops = [qeye(2) for _ in range(N)]
            ops[site - 1] = op

            Q = tensor(*ops)
            Q_reordered = np.array(Q.full())[index_map][:, index_map]

            Q_list.append(Q_reordered)

        return Q_list, labels

    # ==========================================================
    # Utility functions (kept as methods)
    # ==========================================================

    @staticmethod
    def direct_sum(matrix, N):
        blocks = [matrix] * N
        return np.block([
            [blocks[i] if i == j else np.zeros_like(matrix)
             for j in range(N)]
            for i in range(N)
        ])

    @staticmethod
    def pad_vector_with_zeros(vector, N):
        D = len(vector)
        padded = np.zeros(D * N)
        padded[:D] = vector
        return padded
