import numpy as np
from qutip import (
    Qobj, basis, liouvillian,
    spre, spost
)
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath


class HEOMEngine:
    """
    HEOM solver wrapper for third-order spectroscopy.
    """

    def __init__(self, system, NC=3, Nk=1):

        self.system = system
        self.NC = NC
        self.Nk = Nk

        self.Hsys = Qobj(system.ham_sys)
        self.nsite = system.nsite
        self.nbath = system.nbath

        self.mu_p = np.tril(system.dipole, 0)
        self.mu_m = np.triu(system.dipole, 0)

        self.O_list = [Qobj(q) for q in system.ham_sysbath]

        self.baths, self.terminator = self._build_baths()

        self.options = {
            "nsteps": 15000,
            "store_ados": True,
            "progress_bar": False
        }

        self.N_ADO = self._compute_n_ados()
        self.rho0 = self._build_initial_ado_state()

    # ==========================================================
    # Bath construction
    # ==========================================================

    def _build_baths(self):

        baths = []
        terminator = []

        for op in self.O_list:
            bath = DrudeLorentzBath(
                op,
                self.system.lam,
                self.system.gamma,
                self.system.kT,
                self.Nk
            )
            baths.append(bath)
            terminator.append(bath.terminator()[1])

        return baths, terminator

    # ==========================================================
    # ADO initialization
    # ==========================================================

    def _compute_n_ados(self):

        HL = liouvillian(self.Hsys)
        solver = HEOMSolver(
            HL,
            self.baths,
            max_depth=self.NC,
            options={"store_ados": True}
        )

        L = solver._calculate_rhs()(0).full()
        return len(L) // (self.nsite**2)

    def _build_initial_ado_state(self):

        rho0 = basis(self.nsite, 0) * basis(self.nsite, 0).dag()
        rho_list = [rho0.full()]

        for _ in range(1, self.N_ADO):
            rho_list.append(np.zeros_like(rho0.full()))

        return np.array(rho_list)

    # ==========================================================
    # HEOM propagation
    # ==========================================================

    def evolve(self, rho, tf, dt):

        HL = liouvillian(self.Hsys)
        for term in self.terminator:
            HL += term

        solver = HEOMSolver(
            HL,
            self.baths,
            max_depth=self.NC,
            options=self.options
        )

        tlist = np.arange(0, tf + dt, dt) / 5308.0

        result = solver.run(rho, tlist)

        ado_state = result.ado_states[-1]

        ados = [
            ado_state.extract(label).full()
            for label in ado_state.filter()
        ]

        return np.array(ados)

    # ==========================================================
    # Light-matter interaction
    # ==========================================================

    def _strike(self, rho, t, direction, pm):

        if pm == "+":
            mu = self.mu_p
        else:
            mu = self.mu_m

        if direction == "left":
            rho_new = [mu @ r for r in rho]
        else:
            rho_new = [r @ mu for r in rho]

        return self.evolve(np.array(rho_new), t, dt=5.0)

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

        rho_final = rho[0].reshape(self.nsite, self.nsite)

        return np.trace(self.mu_m @ rho_final)

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
    # Fourier Transform
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

        exp1 = np.exp(1j*(1/self.system.hbar)
                      * np.outer(energy1s, times))

        exp3 = np.exp(1j*(1/self.system.hbar)
                      * np.outer(energy3s, times))

        spectrum = (
            np.einsum('ws,xu,uts->xtw', exp1, exp3, Rnr).real
            + np.einsum('ws,xu,uts->xtw',
                        exp1.conj(), exp3, Rrp).real
        )

        spectra = [spectrum[:, t2, :] for t2 in range(len(time2s))]

        return energy1s, energy3s, time2s, spectra 
