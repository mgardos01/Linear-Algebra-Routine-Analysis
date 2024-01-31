# AM 148: Final Project Report

## Overview
Explores the implementation of linear algebra routines optimized for GPU computing using the HIP C++ API, focused on enhancing computational efficiency for large datasets in scientific computing.

## Highlights
- Developed implementations of the common `dasum`, `dnrm2`, `dgemm`, `dcopy`, and `daxpy` routines, aiming to optimize performance and tackle limitations of existing algorithms.
- Utilized parallel reduction and grid-stride loops for better performance compared to their cuBlas equivalents, with specific enhancements for large data handling.
- Conducted comprehensive benchmarks to compare these custom implementations with their cuBlas counterparts, demonstrating improvements and identifying areas for future optimization.

## Running the Code
Instructions are provided for compiling and executing the test suite on Lux's slurm queue or a local HIP runtime, along with steps to visualize results using a Python script.

For more details on implementation and testing methodologies, refer to the full report.
