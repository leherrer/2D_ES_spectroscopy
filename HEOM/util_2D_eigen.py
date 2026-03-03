import numpy as np
import time
from numpy import linalg as LA
from qutip import Qobj, spre, spost
from concurrent.futures import ThreadPoolExecutor
import scipy.linalg as sp

class LiouvilleEigenEngine:
    """
    Diagonal Liouville-space spectroscopy engine for HEOM.
    Computes 2D response using eigen-decomposition of the full Liouvillian.
    """

    def __init__(self, heom_engine):
        self.heom = heom_engine
        self.Nsystem = heom_engine.nsite
        self.N_ADO = heom_engine.N_ADO
        self.hbar = heom_engine.system.hbar

        # Build Liouvillian
        self._build_liouville_matrix()
        # Build ADO dipole superoperators
        self._build_dipole_superoperators()
        # Diagonalize Liouvillian
        self._diagonalize_liouvillian()

    # ==========================================================
    # Build Liouvillian matrix
    # ==========================================================
    def _build_liouville_matrix(self):
        """
        Build the full HEOM Liouville-space matrix including terminators.
        """

        from qutip import liouvillian
        from qutip.solver.heom import HEOMSolver

        # --- 1️⃣ System Hamiltonian Liouvillian ---
        Hsys = self.heom.Hsys  # Qobj
        L = liouvillian(Hsys)

        # --- 2️⃣ Add terminators if available ---
        baths = getattr(self.heom, "baths", None)
        terminator = getattr(self.heom, "terminator", None)
        if baths is None:
            raise ValueError("HEOMEngine has no baths attribute.")

        if terminator is not None:
            for term in terminator:
                L += term

        # --- 3️⃣ Build HEOM solver to access full Liouvillian ---
        solver = HEOMSolver(L, baths, max_depth=self.heom.NC, options=getattr(self.heom, "options", {}))

        # --- 4️⃣ Extract full Liouvillian matrix ---
        L_matrix_qobj = solver._calculate_rhs()(0)   # returns Qobj
        self.L_matrix = L_matrix_qobj.full(order="C")  # convert to numpy array

        # Optional: print size for sanity check
        print("Liouville matrix shape:", self.L_matrix.shape)

    # ==========================================================
    # Build dipole superoperators
    # ==========================================================
    def _build_dipole_superoperators(self):
        """
        Build all dipole superoperators and vectors for HEOM + ADOs.
        """

        from qutip import Qobj, spre, spost
        SystemHamiltonian = self.heom.system.__class__  # access pad_vector_with_zeros

        # -------------------------------
        # 1️⃣ Basic dipole operators
        # -------------------------------
        muplus = Qobj(self.heom.mu_p)
        muminus = Qobj(self.heom.mu_m)

        # Superoperators in Liouville space
        mu_plus_s = (spre(muplus) - spost(muplus)).full(order="C")
        mu_minus_s = (spre(muminus) - spost(muminus)).full(order="C")

        # Left/right superoperators
        mu_plus_s_left  = spre(muplus).full(order="C")
        mu_plus_s_right = spost(muplus).full(order="C")
        mu_minus_s_left  = spre(muminus).full(order="C")
        mu_minus_s_right = spost(muminus).full(order="C")

        # -------------------------------
        # 2️⃣ Vectorized operators and rho0
        # -------------------------------
        Nsystem = self.Nsystem
        N_ADO = self.N_ADO

        vec_rho0 = self.heom.rho0.reshape(N_ADO * Nsystem**2, order="C")
        mu_plus_vec  = muplus.full(order="C").reshape(Nsystem**2, order="C")
        mu_minus_vec = muminus.full(order="C").reshape(Nsystem**2, order="C")
        mu_plus_vec_left  = muplus.dag().full(order="C").reshape(Nsystem**2, order="C")
        mu_minus_vec_left = muminus.dag().full(order="C").reshape(Nsystem**2, order="C")

        # -------------------------------
        # 3️⃣ ADO superoperators
        # -------------------------------
        self.Ado_mu_plus_s  = self._direct_sum(mu_plus_s)
        self.Ado_mu_minus_s = self._direct_sum(mu_minus_s)

        self.Ado_mu_plus_s_left  = self._direct_sum(mu_plus_s_left)
        self.Ado_mu_plus_s_right = self._direct_sum(mu_plus_s_right)
        self.Ado_mu_minus_s_left = self._direct_sum(mu_minus_s_left)
        self.Ado_mu_minus_s_right= self._direct_sum(mu_minus_s_right)

        # -------------------------------
        # 4️⃣ ADO vectors
        # -------------------------------
        self.Ado_mu_plus_vec       = np.tile(mu_plus_vec, N_ADO)
        self.Ado_mu_minus_vec      = np.tile(mu_minus_vec, N_ADO)
        self.Ado_mu_plus_vec_left  = np.tile(mu_plus_vec_left, N_ADO)
        self.Ado_mu_minus_vec_left = np.tile(mu_minus_vec_left, N_ADO)

        # -------------------------------
        # 5️⃣ Expected value vectors (padded)
        # -------------------------------
        self.Ado_mu_plus_vec_exp       = SystemHamiltonian.pad_vector_with_zeros(mu_plus_vec, N_ADO)
        self.Ado_mu_minus_vec_exp      = SystemHamiltonian.pad_vector_with_zeros(mu_minus_vec, N_ADO)
        self.Ado_mu_plus_vec_left_exp  = SystemHamiltonian.pad_vector_with_zeros(mu_plus_vec_left, N_ADO)
        self.Ado_mu_minus_vec_left_exp = SystemHamiltonian.pad_vector_with_zeros(mu_minus_vec_left, N_ADO)

        # -------------------------------
        # 6️⃣ Store vec_rho0
        # -------------------------------
        self.vec_rho0 = vec_rho0

        print("Dipole superoperators and vectors constructed.")

    def _direct_sum(self, matrix):
        """Direct sum of a matrix over all ADOs"""
        blocks = [matrix] * self.N_ADO
        return np.block([
            [blocks[i] if i == j else np.zeros_like(matrix)
             for j in range(self.N_ADO)]
            for i in range(self.N_ADO)
        ])

    # ==========================================================
    # Diagonalize Liouvillian
    # ==========================================================
    def _diagonalize_liouvillian(self):
        print("Diagonalizing Liouvillian...")
        start = time.time()
        eigenvalues, eigenvectors = LA.eig(self.L_matrix)
        self.diagonal_matrix = np.diag(eigenvalues)
        self.U = eigenvectors
        self.Uinv = LA.inv(eigenvectors)
        print(f"Done in {time.time() - start:.2f} s")

    def _response_element(self, t1, t2, t3):
        """
        Compute a single 2D response element using full HEOM Liouvillian.
        """

        # --- Exponentials of diagonalized Liouvillian ---
        exp1 = np.diag(np.exp(np.diag(self.diagonal_matrix) * t1 / self.hbar))
        exp2 = np.diag(np.exp(np.diag(self.diagonal_matrix) * t2 / self.hbar))
        exp3 = np.diag(np.exp(np.diag(self.diagonal_matrix) * t3 / self.hbar))


        vec = self.Ado_mu_minus_s @ self.vec_rho0
        vec = self.U @ exp1 @ self.Uinv @ vec
        vec = self.Ado_mu_plus_s @ vec
        vec = self.U @ exp2 @ self.Uinv @ vec
        vec = self.Ado_mu_plus_s @ vec
        vec = self.U @ exp3 @ self.Uinv @ vec
        rp = self.Ado_mu_plus_vec_left_exp @ vec

        # --- Non-Rephasing ---
        vec = self.Ado_mu_plus_s @ self.vec_rho0
        vec = self.U @ exp1 @ self.Uinv @ vec
        vec = self.Ado_mu_minus_s @ vec
        vec = self.U @ exp2 @ self.Uinv @ vec
        vec = self.Ado_mu_plus_s @ vec
        vec = self.U @ exp3 @ self.Uinv @ vec
        nr = self.Ado_mu_plus_vec_left_exp @ vec

        return rp, nr

    # ==========================================================
    # Parallel 2D response
    # ==========================================================
    def compute_R_signal_parallel(self, time2s, time_final, dt, ncores=4):
        times = np.arange(0.0, time_final, dt)
        R_rp = np.zeros((len(times), len(time2s), len(times)), dtype=complex)
        R_nr = np.zeros((len(times), len(time2s), len(times)), dtype=complex)
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            futures = []
            for i, t3 in enumerate(times):
                for j, t2 in enumerate(time2s):
                    for k, t1 in enumerate(times):
                        futures.append(
                            executor.submit(
                                self._response_element, t1, t2, t3
                            )
                        )

            idx = 0
            for i in range(len(times)):
                for j in range(len(time2s)):
                    for k in range(len(times)):
                        rp, nr = futures[idx].result()
                        R_rp[i, j, k] = rp
                        R_nr[i, j, k] = nr
                        idx += 1

        return R_rp, R_nr

    # ==========================================================
    # Fourier transform
    # ==========================================================
    # def fourier_transform(self, R, e1_range, e3_range, time2s, time_final, dt):
    #     e1_min, e1_max, de1 = e1_range
    #     e3_min, e3_max, de3 = e3_range
    #     energy1s = np.arange(e1_min, e1_max, de1)
    #     energy3s = np.arange(e3_min, e3_max, de3)
    #     times = np.arange(0.0, time_final, dt)
    #
    #     exp1 = np.exp(1j * (1 / self.hbar) * np.outer(energy1s, times))
    #     exp3 = np.exp(1j * (1 / self.hbar) * np.outer(energy3s, times))
    #
    #     spectrum = np.einsum('ws,xu,uts->xtw', exp1, exp3, R).real
    #     spectra = [spectrum[:, t2, :] for t2 in range(len(time2s))]
    #     return energy1s, energy3s, spectra


    def fourier_transform(self, Rrp, Rnr ,e1_range, e3_range, time2s, time_final, dt):

        e1_min, e1_max, de1 = e1_range
        e3_min, e3_max, de3 = e3_range
        omega1s = np.arange(e1_min, e1_max, de1)
        omega3s = np.arange(e3_min, e3_max, de3)
        times = np.arange(0.0, time_final, dt)

        spectrum = np.zeros( (len(omega3s),len(time2s),len(omega1s)) )

        Rsignal = []
        Rsignal.append(Rrp)
        Rsignal.append(Rnr)

        expi1 = np.exp(1j*(1/self.hbar)*np.outer(omega1s,times))
        expi1[:,0] *= 0.5*dt
        expi1[:,1:] *= dt
        expi3 = np.exp(1j*(1/self.hbar)*np.outer(omega3s,times))
        expi3[:,0] *= 0.5*dt
        expi3[:,1:] *= dt
        spectrum =  np.einsum('ws,xu,uts->xtw',expi1,expi3,Rnr).real
        spectrum += np.einsum('ws,xu,uts->xtw',expi1.conj(),expi3,Rrp).real


        print("done.")

        spectra = []
        for t2 in range(len(time2s)):
            spectra.append( spectrum[:,t2,:] )

        return omega1s, omega3s, spectra
        #return energy1s, energy3s, time2s, spectra, times, Rsignal






    # ==========================================================
    # Normalize intensities
    # ==========================================================
    @staticmethod
    def normalize_intensities(intensities):
        min_val = intensities.min()
        max_val = intensities.max()
        out = np.zeros_like(intensities, dtype=float)
        pos = intensities > 0
        neg = intensities < 0
        out[pos] = intensities[pos] / max_val
        out[neg] = intensities[neg] / -min_val
        return out
