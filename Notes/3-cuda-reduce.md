# Cuda reduce Naive

$ ./laplace 
Jacobi relaxation Calculation: 4096 x 4096 mesh

|       | Serial | CUDA swap|CUDA reduce|
|---:   | ---:   | ---:     | ---:     |
|   0 | 0.250000 | 0.250000 | 0.250000 |
|  100| 0.002397 | 0.002397 | 0.002397 |
|  200| 0.001204 | 0.001204 | 0.001204 |
|  300| 0.000804 | 0.000804 | 0.000804 |
|  400| 0.000603 | 0.000603 | 0.000603 |
|  500| 0.000483 | 0.000483 | 0.000483 |
|  600| 0.000403 | 0.000403 | 0.000403 |
|  700| 0.000345 | 0.000345 | 0.000345 |
|  800| 0.000302 | 0.000302 | 0.000302 |
|  900| 0.000269 | 0.000269 | 0.000269 |
| total| 257.286935 s | 281.425898 s | 74.529034 s |

### nvprof results

```text
$ sudo nvprof ./laplace 
==44784== NVPROF is profiling process 44784, command: ./laplace
Jacobi relaxation Calculation: 4096 x 4096 mesh
    0, 0.250000
  100, 0.002397
  200, 0.001204
  300, 0.000804
  400, 0.000603
  500, 0.000483
  600, 0.000403
  700, 0.000345
  800, 0.000302
  900, 0.000269
 total: 73.404252 s
==44784== Profiling application: ./laplace
==44784== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.84%  28.3372s      1000  28.337ms  28.115ms  50.681ms  reduce0(double*, double*, int, int, double*)
                   31.10%  22.6914s      1000  22.691ms  22.650ms  30.446ms  stencil_cu(double*, double*, int, int)
                   29.11%  21.2409s      1000  21.241ms  21.202ms  30.945ms  swap_cu(double*, double*, int, int)
                    0.53%  384.76ms         4  96.191ms  86.028ms  111.04ms  [CUDA memcpy HtoD]
                    0.43%  313.16ms      1000  313.16us  311.52us  324.86us  [CUDA memcpy DtoH]
                    0.00%  4.9920us         1  4.9920us  4.9920us  4.9920us  initialize_cu(double*, double*, int, int)
      API calls:   98.02%  72.2993s      4001  18.070ms  1.6150us  50.687ms  cudaDeviceSynchronize
                    1.45%  1.07045s      1004  1.0662ms  635.69us  110.14ms  cudaMemcpy
                    0.42%  309.65ms      1002  309.03us  9.7080us  185.29ms  cudaMalloc
                    0.08%  56.748ms      3001  18.909us  8.7230us  1.3143ms  cudaLaunchKernel
                    0.02%  18.176ms         2  9.0879ms  335.44us  17.840ms  cudaFree
                    0.00%  942.12us      3001     313ns     145ns  3.2200us  cudaGetLastError
                    0.00%  468.44us        97  4.8290us     354ns  196.53us  cuDeviceGetAttribute
                    0.00%  211.74us         1  211.74us  211.74us  211.74us  cuDeviceTotalMem
                    0.00%  74.581us         1  74.581us  74.581us  74.581us  cuDeviceGetName
                    0.00%  9.2790us         1  9.2790us  9.2790us  9.2790us  cuDeviceGetPCIBusId
                    0.00%  4.3140us         3  1.4380us     347ns  3.1050us  cuDeviceGetCount
                    0.00%  2.4180us         2  1.2090us     501ns  1.9170us  cuDeviceGet
                    0.00%     564ns         1     564ns     564ns     564ns  cuDeviceGetUuid
```