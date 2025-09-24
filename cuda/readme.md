# compile
```
$ nvcc -Xcompiler -fopenmp -arch=sm_89 jacobi.cu laplace2d.c -o laplace
```