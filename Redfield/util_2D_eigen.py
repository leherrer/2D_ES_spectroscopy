import numpy as np
import time
from numpy import linalg as LA
from qutip import Qobj, spre, spost, liouvillian


class LiouvilleEigenEngineRedfield:
    """
    Diagonal Liouville-space spectroscopy engine for Redfield.
    Uses eigen-decomposition of Bloch-Redfield Liouvillian.
    """

    def __init__(self, redfield_engine):

        self.rd = redfield_engine
        self.Nsystem = redfield_engine.nsite
        self.hbar = redfield_engine.system.hbar

        # Build Liouvillian
        self._build_liouville_matrix()

        # Build dipole superoperators
        self._build_dipole_superoperators()

        # Diagonalize
        self._diagonalize_liouvillian()

    # ==========================================================
    # Build Redfield Liouvillian
    # ==========================================================

    def _build_liouville_matrix(self):
        """
        Build full Bloch-Redfield Liouvillian matrix.
        """

        print("Building Redfield Liouvillian...")

        # Hamiltonian part
        #L = liouvillian(self.rd.Hsys)

        # Add Redfield dissipator tensor
        L = self.rd.RD

        self.L_matrix = L.full(order="C")

        print("Liouville matrix shape:", self.L_matrix.shape)

    # ==========================================================
    # Build dipole superoperators (system only)
    # ==========================================================

    def _build_dipole_superoperators(self):

        muplus = Qobj(self.rd.mu_p)
        muminus = Qobj(self.rd.mu_m)

        # Commutator superoperators
        self.mu_plus_s  = (spre(muplus) - spost(muplus)).full(order="C")
        self.mu_minus_s = (spre(muminus) - spost(muminus)).full(order="C")

        # Left-only superoperator (for expectation value)
        self.mu_plus_left = spre(muplus).full(order="C")

        # Vectorized operators
        self.mu_plus_vec_left = (
            muplus.dag().full(order="C")
            .reshape(self.Nsystem**2, order="C")
        )

        # Vectorized rho0
        self.vec_rho0 = (
            self.rd.rho0.full(order="C")
            .reshape(self.Nsystem**2, order="C")
        )

        print("Dipole superoperators constructed.")

    # ==========================================================
    # Diagonalize Liouvillian
    # ==========================================================

    def _diagonalize_liouvillian(self):

        print("Diagonalizing Liouvillian...")
        start = time.time()

        eigenvalues, eigenvectors = LA.eig(self.L_matrix)

        self.evals = eigenvalues
        self.U = eigenvectors
        self.Uinv = LA.inv(eigenvectors)

        print(f"Done in {time.time() - start:.2f} s")

    # ==========================================================
    # Single response element
    # ==========================================================

    def _propagate(self, vec, t):

        exp_diag = np.exp(self.evals * t / self.hbar)
        return self.U @ (exp_diag * (self.Uinv @ vec))


    def _response_element(self, t1, t2, t3):

        # =========================
        # Rephasing
        # =========================
        vec = self.mu_minus_s @ self.vec_rho0
        vec = self._propagate(vec, t1)

        vec = self.mu_plus_s @ vec
        vec = self._propagate(vec, t2)

        vec = self.mu_plus_s @ vec
        vec = self._propagate(vec, t3)

        rp = self.mu_plus_vec_left @ vec

        # =========================
        # Non-rephasing
        # =========================
        vec = self.mu_plus_s @ self.vec_rho0
        vec = self._propagate(vec, t1)

        vec = self.mu_minus_s @ vec
        vec = self._propagate(vec, t2)

        vec = self.mu_plus_s @ vec
        vec = self._propagate(vec, t3)

        nr = self.mu_plus_vec_left @ vec

        return rp, nr

    # ==========================================================
    # Full 2D response
    # ==========================================================

    def compute_R_signal(self, time2s, time_final, dt):

        times = np.arange(0.0, time_final, dt)

        R_rp = np.zeros((len(times), len(time2s), len(times)), dtype=complex)
        R_nr = np.zeros_like(R_rp)

        for i, t3 in enumerate(times):
            for j, t2 in enumerate(time2s):
                for k, t1 in enumerate(times):

                    rp, nr = self._response_element(t1, t2, t3)

                    R_rp[i, j, k] = rp
                    R_nr[i, j, k] = nr

        return R_rp, R_nr

    # ==========================================================
    # Fourier Transform
    # ==========================================================

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
