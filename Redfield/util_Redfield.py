import numpy as np
from qutip import (
    Qobj, basis,
    brmesolve, bloch_redfield_tensor
)


class RedfieldEngine:
    """
    Bloch-Redfield solver wrapper for third-order spectroscopy.
    Structure mirrors HEOMEngine.
    """

    def __init__(self, system, sec_cutoff=0.001):

        self.system = system
        self.sec_cutoff = sec_cutoff

        self.nsite = system.nsite
        self.nbath = system.nbath

        # Hamiltonian
        self.Hsys = Qobj(system.ham_sys)

        # Dipoles
        self.mu_p = np.tril(system.dipole, 0)
        self.mu_m = np.triu(system.dipole, 0)

        # Bath operators
        self.O_list = [Qobj(q) for q in system.ham_sysbath]

        # Initial density matrix
        self.rho0 = basis(self.nsite, 0) * basis(self.nsite, 0).dag()

        self.options = {
            "nsteps": 15000,
            "progress_bar": False
        }

        # Pre-build Redfield tensor (optional, for analysis)
        self.RD = self._build_redfield_tensor()

    # ==========================================================
    # Spectral density (Drude-Lorentz)
    # ==========================================================

    def _calculate_DL(self, w):

        lam = self.system.lam
        gamma = self.system.gamma
        b = 1.0 / self.system.kT

        if w == 0:
            return 2 * np.pi * 2.0 * lam / (np.pi * gamma * b)
        else:
            return (
                2 * np.pi
                * (2.0 * lam * gamma * w / (np.pi * (w**2 + gamma**2)))
                * ((1 / (np.exp(w * b) - 1)) + 1)
            )

    # ==========================================================
    # Build Redfield tensor
    # ==========================================================

    def _build_redfield_tensor(self):

        a_ops = [[Q, self._calculate_DL] for Q in self.O_list]

        RD = bloch_redfield_tensor(
            self.Hsys,
            a_ops=a_ops,
            sec_cutoff=self.sec_cutoff,
            fock_basis=True
        )

        return RD

    # ==========================================================
    # Time evolution
    # ==========================================================

    def evolve(self, rho, tf, dt):

        tlist = np.arange(0, tf + dt, dt) / 5308.0

        a_ops = [[Q, self._calculate_DL] for Q in self.O_list]

        result = brmesolve(
            self.Hsys,
            rho,
            tlist,
            a_ops=a_ops,
            options=self.options
        )

        return result.states[-1]

    # ==========================================================
    # Light-matter interaction
    # ==========================================================

    def _strike(self, rho, t, direction, pm):

        if pm == "+":
            mu = Qobj(self.mu_p)
        else:
            mu = Qobj(self.mu_m)

        if direction == "left":
            rho_new = mu * rho
        else:
            rho_new = rho * mu

        return self.evolve(rho_new, t, dt=5.0)

    # ==========================================================
    # Third-order response
    # ==========================================================

    def compute_response(self, t1, t2, t3, pathway):

        rho = self.rho0.copy()

        pathways = {
            "1": [("right","-"), ("left","+"), ("right","+")],
            "2": [("right","-"), ("right","+"), ("left","+")],
            "3": [("right","-"), ("left","+"), ("left","+")],
            "4": [("left","+"), ("right","-"), ("right","+")],
            "5": [("left","+"), ("left","-"), ("left","+")],
            "6": [("left","+"), ("right","-"), ("left","+")]
        }

        seq = pathways[pathway]

        rho = self._strike(rho, t1, *seq[0])
        rho = self._strike(rho, t2, *seq[1])
        rho = self._strike(rho, t3, *seq[2])

        return (Qobj(self.mu_m) * rho).tr()

    # ==========================================================
    # Full 2D signal
    # ==========================================================

    def compute_R_signal(self,
                         time2_min, time2_max, dt2,
                         time_final, dt):

        time2s = np.arange(time2_min, time2_max + 1e-8, dt2)
        times = np.arange(0.0, time_final, dt)

        Rrp = np.zeros((len(times), len(time2s), len(times)), dtype=complex)
        Rnr = np.zeros_like(Rrp)

        for i, t3 in enumerate(times):
            for j, t2 in enumerate(time2s):
                for k, t1 in enumerate(times):

                    phi = [
                        self.compute_response(t1,t2,t3,str(p))
                        for p in range(1,7)
                    ]

                    Rrp[i,j,k] = phi[0] + phi[1] - phi[2]
                    Rnr[i,j,k] = phi[3] + phi[4] - phi[5]

        return Rrp, Rnr

    # ==========================================================
    # Fourier transform
    # ==========================================================

    def fourier_transform(self,
                          Rrp, Rnr,
                          e1_range, e3_range,
                          time2_min, time2_max, dt2,
                          time_final, dt):

        e1_min, e1_max, de1 = e1_range
        e3_min, e3_max, de3 = e3_range

        energy1s = np.arange(e1_min, e1_max, de1)
        energy3s = np.arange(e3_min, e3_max, de3)

        time2s = np.arange(time2_min, time2_max + 1e-8, dt2)
        times = np.arange(0.0, time_final, dt)

        exp1 = np.exp(
            1j*(1/self.system.hbar)
            * np.outer(energy1s, times)
        )

        exp3 = np.exp(
            1j*(1/self.system.hbar)
            * np.outer(energy3s, times)
        )

        spectrum = (
            np.einsum('ws,xu,uts->xtw', exp1, exp3, Rnr).real
            + np.einsum('ws,xu,uts->xtw',
                        exp1.conj(), exp3, Rrp).real
        )

        spectra = [spectrum[:, t2, :] for t2 in range(len(time2s))]

        return energy1s, energy3s, time2s, spectra
