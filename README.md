# HEOM 2D Spectroscopy – Liouville Eigenbasis Implementation

This repository implements **2D electronic spectroscopy simulations** using the **Hierarchical Equations of Motion (HEOM)** formalism and a **Liouville-space diagonalization approach** for efficient response function evaluation.

The code is written in an object-oriented structure and separates:

* System Hamiltonian construction
* HEOM hierarchy generation
* Liouville-space diagonalization
* 2D response calculation
* Fourier transformation to frequency domain

---

## 📌 Features

* System–bath Hamiltonian construction
* HEOM hierarchy generation (finite depth and Matsubara terms)
* Full Liouvillian diagonalization
* Parallel computation of response functions
* 2D Fourier transform
* Export of raw response and frequency-domain spectra

---

## 📂 Project Structure

```
.
├── main.py                 # Main execution script
├── util_HAM.py             # System Hamiltonian class
├── util_HEOM.py            # HEOM engine
├── util_2D_eigen.py        # Liouville eigenbasis engine
```

---

## ⚙️ Requirements

* Python ≥ 3.9
* NumPy
* Multiprocessing (standard library)


## 🚀 How to Run

```bash
python main.py
```

**Important:**
The script is protected with:

```python
if __name__ == "__main__":
    main()
```

This is **critical** for multiprocessing to work correctly.

---

## 🔬 Simulation Parameters

### Physical Parameters

| Parameter | Meaning               | Units |
| --------- | --------------------- | ----- |
| `lam`     | Reorganization energy | cm⁻¹  |
| `tau_c`   | Bath correlation time | fs    |
| `T`       | Temperature           | K     |
| `J`       | Electronic coupling   | cm⁻¹  |
| `mu`      |Dipole moment          | a.u.  |

Constants used:

* kB = 0.69352 cm⁻¹/K
* hbar = 5308.8 cm⁻¹·fs

---

### HEOM Parameters

| Parameter | Meaning                         |
| --------- | ------------------------------- |
| `NC`      | Hierarchy depth                 |
| `Nk`      | Number of Matsubara frequencies |

---

### Time Grid

| Parameter | Meaning                     |
| --------- | --------------------------- |
| `t_final` | Final propagation time (fs) |
| `dt`      | Time step (fs)              |
| `Time2s`  | Population times t₂         |

---

### Frequency Window

```python
e1_range = (-500.0, 500.0, 5.0)
e3_range = (-500.0, 500.0, 5.0)
```

Format:

```
(min, max, step)  # cm^-1
```

---

## 🏗 Workflow Overview

### 1️⃣ Build System Hamiltonian

Constructed using:

```python
SystemHamiltonian(...)
```

Includes:

* Electronic Hamiltonian
* Dipole operator
* System–bath coupling
* Temperature and bath parameters

---

### 2️⃣ Build HEOM Engine

```python
HEOMEngine(system, NC, Nk)
```

Generates:

* ADO hierarchy
* HEOM Liouvillian

---

### 3️⃣ Diagonalize Liouvillian

```python
LiouvilleEigenEngine(heom)
```

Performs:

* Full diagonalization
* Eigenbasis propagation setup

---

### 4️⃣ Compute Time-Domain Response

```python
compute_R_signal_parallel(...)
```

* Parallelized over multiple cores
* Returns time-domain response functions

---

### 5️⃣ Save Raw Response

Saved as:

```
Rsignal_HEOM_J-XXX_l-XXX.npy
```

---

### 6️⃣ Fourier Transform

```python
fourier_transform(...)
```

Computes:

* ω₁
* ω₃
* 2D spectra

---

### 7️⃣ Save 2D Spectra

Output format:

```
omega1   omega3   Re[S(ω3, ω1)]
```

Filename example:

```
2d_t2-0.0_HEOM_dt-10_tf-500_L-3_K-1_tau-100.0_l-60.0.dat
```

---

## 📊 Output Files

### Raw Time-Domain Signal

```
Rsignal_HEOM_J-100.0_l-60.0.npy
```

---

### Frequency-Domain 2D Spectra

```
2d_t2-0.0_HEOM_dt-10_tf-500_L-3_K-1_tau-100.0_l-60.0.dat
```

Format:

```
ω1  ω3  Spectrum(ω3, ω1)
```

Blank line separates ω1 slices.

---

## ⚡ Parallelization

The response function is computed using:

```python
ncores=5
```

Adjust according to your machine:

```python
import multiprocessing
ncores = multiprocessing.cpu_count()
```

---

## 📈 Example System

The current script runs a two-site excitonic dimer:

```python
ham_sys_x = [[ -50, J ],
             [  J, 50 ]]
```

With asymmetric dipole moments:

```python
dipole_x = [1.0, -0.2]
```

---

## 🧪 Performance Notes

* Liouvillian diagonalization scales poorly with hierarchy size.
* Increasing `NC` significantly increases computational cost.
* For large systems, memory usage can become substantial.

---

## 👤 Author

Luis E. Herrera Rodriguez
