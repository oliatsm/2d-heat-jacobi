# 2D Heat Plate Simulation on GPU

This project implements a GPU-accelerated simulation of heat distribution on a 2D metal plate using the Jacobi iterative method.

The serial implementation is based on the [OpenACC Training Materials](https://github.com/OpenACC/openacc-training-materials) provided by OpenACC.org  
This repository extends the original tutorial by adding a CUDA version for GPU acceleration.

---


## Problem Description

Heat diffusion on a 2D plate is modeled by solving the Laplace equation iteratively:

$$
A_{j,i}^{\text{new}} = \tfrac{1}{4}\Big( A_{j,i+1} + A_{j,i-1} + A_{j-1,i} + A_{j+1,i} \Big)
$$


- **Boundary conditions:**
    - Top row fixed at 1.0 (heat source).
    - Other boundaries fixed at 0.0.

- **Convergence:**
    Iteration stops when the maximum difference between iterations falls below a tolerance, or after a maximum number of iterations.

---

## Project Structure

- `serial/` – serial implementation in C.

- `cuda/` – GPU implementation in CUDA.

---

## Performance

|Grid Size  | Serial    | CUDA     | Speedup|
| ---       | ---:      | ---:     | ---:   |
|1024 x 1024|   6.6511 s| 0.0757 s | 88x    |
|2048 x 2048|  26.7749 s| 0.4801 s | 55x    |
|4096 x 4096| 154.8857 s| 2.4873 s | 62x    | 
|8192 x 8192| 437.6124 s| 7.7569 s | 56x    |

### Environment

- CPU: AMD Ryzen 7 5800X 8-Core Processor
- GPU: NVIDIA GeForce RTX 4070 Ti
- Ubuntu 22.04.1

---
## Build & Run

### Requirements

- **gcc** compiler for serial version.

- [**NVIDIA CUDA Toolkit**](https://developer.nvidia.com/cuda-toolkit) - **nvcc** compiler for CUDA version.

- **Linux** environment recommended.

### Serial version

```
cd serial
make
./jacobi

```

### CUDA version 

```
cd cuda
make
./jacobi
```
---
