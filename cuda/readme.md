# compile
```
$ nvcc -Xcompiler -fopenmp -arch=sm_89 jacobi.cpp laplace2d.cu -o laplace
```