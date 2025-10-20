Here’s a concise, clean version of your README without emojis or unnecessary formatting:

---

# FEM Conjugate Gradient Solver

This project implements a Finite Element Method (FEM) solver using the Conjugate Gradient (CG) method for sparse linear systems. It includes both CPU and GPU (CUDA) implementations.

---

## File Overview

| File                       | Description                                                                                                                                                                                                          |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **vector.hpp**             | Vector class with functions for setting size, computing norms, and inner products. Includes a complete CPU implementation and four TODOs for the GPU version. Use the flag `-DDISABLE_CUDA` to compile without CUDA. |
| **sparse_matrix.hpp**      | Sparse matrix implementation in Compressed Row Storage (CRS) format. Contains two TODOs for GPU implementation. See lines 291–299 for the CPU matrix-vector product example.                                         |
| **conjugate_gradient.hpp** | Conjugate Gradient solver. Should not be modified but can be reviewed to see which matrix and vector functions are used.                                                                                             |
| **fem_application.cpp**    | Main program for CPU execution. Handles problem setup, solver execution, and output.                                                                                                                                 |
| **fem_application.cu**     | Main program for GPU execution (CUDA). Compiles and runs the solver on GPU.                                                                                                                                          |

---

## Compilation

### CPU version (recommended for local use)

```bash
g++ -O3 -march=native -fopenmp fem_application.cpp -o app.host -DDISABLE_CUDA
```

### GPU version (CUDA)

```bash
nvcc -O3 -arch=sm_75 fem_application.cu -o app.cuda
```

Replace `sm_75` with your GPU’s compute capability if needed.

---

## Running the Code

### On local machine (CPU)

```bash
./app.host -N 128 -repeat 200 -number float
```

Example output:

```
Computing on a 128^3 domain with float numbers
L2 discretization error with sparse matrix: 0.00270006
Mat-vec of size 2097152: 0.016582 seconds or 28.9168 GB/s
Conjugate gradient solve of size 2097152 in 371 iterations: 7.54427 seconds or 103.13 MUPD/s/it
Error conjugate gradient solve: 0.000415534
```

The error should reduce by about a factor of four when doubling `N`. Throughput (MUPD/s/it) enables comparison across problem sizes. The conjugate gradient error will be smaller when using double precision.

---

### On UPPMAX

Compile:

```bash
module load gcc openmpi
g++ -O3 -fopenmp fem_application.cpp -o app.host -DDISABLE_CUDA
```

Run with:

```bash
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=16
./app.host -N 128 -repeat 200 -number float
```



If you only aim for **grade 3 (Pass)**, you can safely **ignore the advanced CELL-C-Sigma format** and other optimization or framework discussions.
Here’s a clear **task list filtered for grade 3**, with what you need to do and what you can skip.

---

# TODO
## Tasks Required for Grade 3

### 1. CPU Performance Tests

* Run the **CPU version** (`fem_application.cpp`) with **OpenMP**.
* Test for:

  ```
  N = 32, 64, 128, 256
  number = float and double
  ```
* Record:

  * L2 discretization error
  * Matrix–vector performance (GB/s)
  * Conjugate Gradient solver performance (MUPD/s/it)

---

### 2. Complete CUDA Implementation (Basic CRS only)

* Implement the **TODO parts** in:

  * `vector.hpp` (GPU sections)
  * `sparse_matrix.hpp` (GPU CRS SpMV)
* Make sure the CUDA code:

  * Compiles and runs correctly.
  * Produces **similar results** (discretization error and CG error) as CPU.
  * Shows that the error decreases when `N` increases.

You **do not** need to implement CELL-C-Sigma or optimize beyond correctness and basic timing.

---

### 3. Read and Understand

* Read the **Kreutzer et al. (2014)** paper:

  * Introduction → overview only
  * Section 2 → skim
  * Section 3 → read carefully (to explain SELL-C-Sigma concept briefly in your report)

You don’t need to implement the format, only understand what it is and why it helps.

---

### 4. Report Content (minimum for grade 3)

Your report should include:

1. **Brief description** of the code structure and CRS format.
2. **CPU and GPU performance tables/graphs** for the runs above.
3. **Short explanation** of performance trends and differences:

   * Why GPU can be faster/slower.
   * How memory bandwidth and parallelism affect CRS SpMV and CG.
4. **Short conceptual comparison** between CRS and SELL-C-Sigma (from Kreutzer).
5. **Basic discussion of data movement** (host↔device) and its cost.

Length: ~3–5 pages is sufficient.

---

## 🚫 Tasks You Can Skip for Grade 3

You do **not** need to:

* Implement or test **CELL-C-Sigma**.
* Compare to **cuBLAS/cuSPARSE**.
* Measure host-device transfer times in detail.
* Do **profiling or advanced analysis** of data locality.
* Explore **Kokkos** or other frameworks.

