import numpy as np
import time

# OOP modules
from util_HAM import SystemHamiltonian
from util_Redfield import RedfieldEngine
from util_2D_eigen import LiouvilleEigenEngineRedfield


def main():

    kB = 0.69352     # cm^-1 / K
    hbar = 5308.8    # cm^-1 * fs

    # ==========================================================
    # Parameters
    # ==========================================================

    # Physical parameters
    lam = 60.0       # cm^-1
    tau_c = 100.0    # fs
    T = 77.0         # K

    # Time parameters
    t_final = 500.0
    dt = 10.0
    Time2s = [0]

    # Fourier windows
    e1_range = (-400.0, 400.0, 5.0)
    e3_range = (-400.0, 400.0, 5.0)

    # ==========================================================
    # 1️⃣ Build system Hamiltonian
    # ==========================================================

    J = -100.0

    ham_sys_x = np.array([
        [-50., J],
        [J,  50.]
    ])

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
    print("Hamiltonian:", system.ham_sys)
    print("Dipoles:", system.dipole)
    print("System bath coupling:", system.ham_sysbath)
    print("Labels:", system.labels_sysbath)

    # ==========================================================
    # 2️⃣ Build Redfield engine
    # ==========================================================

    redfield = RedfieldEngine(system=system)

    print("Redfield tensor dimension:",
          redfield.RD.shape)

    # ==========================================================
    # 3️⃣ Build diagonal Liouville engine
    # ==========================================================

    eigen_engine = LiouvilleEigenEngineRedfield(redfield)

    # ==========================================================
    # 4️⃣ Compute 2D response
    # ==========================================================

    single_run = eigen_engine._response_element(0, 0, 0)
    print("Single response element:", single_run)

    single_run = eigen_engine._response_element(100, 0, 0)
    print("Single response element:", single_run)
    single_run = eigen_engine._response_element(0, 100, 0)
    print("Single response element:", single_run)
    single_run = eigen_engine._response_element(0, 0, 100)
    print("Single response element:", single_run)

    start_time = time.time()

    Rsignal = eigen_engine.compute_R_signal(
        Time2s,
        t_final,
        dt
    )

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # ==========================================================
    # 5️⃣ Save raw response
    # ==========================================================

    np.save(
        f"Rsignal_Redfield_J-{J:.1f}_l-{lam:.1f}.npy",
        Rsignal
    )

    print(Rsignal[0][0,0,0])
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
            f"2d_t2-{t2:.1f}_Redfield_"
            f"dt-{dt:.0f}_tf-{t_final:.0f}_"
            f"tau-{tau_c:.1f}_"
            f"l-{lam:.1f}.dat"
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
# Entry point protection
# ==========================================================

if __name__ == "__main__":
    main()
