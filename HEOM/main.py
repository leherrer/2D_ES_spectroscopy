import numpy as np
import time

# Your new OOP modules
from util_HAM import SystemHamiltonian
from util_HEOM import HEOMEngine
from util_2D_eigen import LiouvilleEigenEngine



def main():

    kB = 0.69352     # cm^-1 / K
    hbar = 5308.8    # cm^-1 * fs

    # ==========================================================
    # Parameters
    # ==========================================================

    # Physical parameters
    lam = 60.0     ##in cm-1
    tau_c = 100.0  ##in fs
    T = 77.0       ##in K

    # HEOM hierarchy
    NC = 3
    Nk = 1

    # Time parameters
    t_final = 500.0
    dt = 10.0
    Time2s = [0]

    # Fourier windows
    e1_range = (-500.0, 500.0, 5.0)
    e3_range = (-500.0, 500.0, 5.0)

    # ==========================================================
    # 1️⃣ Build system Hamiltonian
    # ==========================================================

    J = -100.0

    ham_sys_x = np.array([[ -50., J ],
                          [ J,  50.]])

    dipole_x = np.array([1.0, -0.2])

    system = SystemHamiltonian(
        ham_sys_x=ham_sys_x,
        dipole_x=dipole_x,
        coupling_sites=[1, 2],
        lam=lam,
        gamma=hbar/tau_c,
        temperature=T
    )

    print("System dimension:", system.nsite)

    print("Hamiltonian:" , system.ham_sys)
    print("Dipoles:" , system.dipole)
    print("System bath coupling:" , system.ham_sysbath)
    print("Labels:" , system.labels_sysbath)


    # ==========================================================
    # 2️⃣ Build HEOM engine
    # ==========================================================

    heom = HEOMEngine(
        system=system,
        NC=NC,
        Nk=Nk
    )

    print("Number of ADOs:", heom.N_ADO)

    # ==========================================================
    # 3️⃣ Build diagonal Liouville engine
    # ==========================================================

    eigen_engine = LiouvilleEigenEngine(heom)

    # ==========================================================
    # 4️⃣ Compute 2D response
    # ==========================================================

    single_run = eigen_engine._response_element(0, 0, 0)
    print(single_run)

    start_time = time.time()

    Rsignal = eigen_engine.compute_R_signal_parallel(
        Time2s,
        t_final,
        dt,
        ncores=5
    )

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # ==========================================================
    # 5️⃣ Save raw response
    # ==========================================================

    np.save(
        f"Rsignal_HEOM_J-{J:.1f}_l-{lam:.1f}.npy",
        Rsignal
    )

    # ==========================================================
    # 6️⃣ Fourier transform
    # ==========================================================

    omega1s, omega3s, spectra = eigen_engine.fourier_transform(
        Rsignal[0],
        Rsignal[1],
        e1_range,
        e3_range,
        Time2s,
        t_final,
        dt
    )

    # ==========================================================
    # 7️⃣ Save 2D spectra
    # ==========================================================

    gam = (1 / tau_c) * system.hbar

    for t2, spectrum in zip(Time2s, spectra):

        filename = (
            f"2d_t2-{t2:.1f}_HEOM_"
            f"dt-{dt:.0f}_tf-{t_final:.0f}_"
            f"L-{NC}_K-{Nk}_"
            f"tau-{tau_c:.1f}_l-{lam:.1f}.dat"
            #f"g-{gam:.1f}_l-{lam:.1f}.dat"
        )

        with open(filename, 'w') as f:
            for w1 in range(len(omega1s)):
                for w3 in range(len(omega3s)):
                    f.write(
                        f"{omega1s[w1]:.8f} "
                        f"{omega3s[w3]:.8f} "
                        f"{spectrum[w3, w1]:.8f}\n"
                    )
                f.write("\n")

    print("All files written successfully.")


# ==========================================================
# Entry point protection (CRITICAL for multiprocessing)
# ==========================================================

if __name__ == "__main__":
    main()
