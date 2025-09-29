Jacobi relaxation Calculation: 4096 x 4096 mesh

on NVIDIA GeForce RTX 4070 Ti


|     | Serial  | CUDA       |OpenACC   |
|---: | ---:    | ---:       |---:      |
|   0 | 0.250000|   0.250000 |   0.250000 |
|  100| 0.002397|   0.002397 |   0.002397 |
|  200| 0.001204|   0.001204 |   0.001204 |
|  300| 0.000804|   0.000804 |   0.000804 |
|  400| 0.000603|   0.000603 |   0.000603 |
|  500| 0.000483|   0.000483 |   0.000483 |
|  600| 0.000403|   0.000403 |   0.000403 |
|  700| 0.000345|   0.000345 |   0.000345 |
|  800| 0.000302|   0.000302 |   0.000302 |
|  900| 0.000269|   0.000269 |   0.000269 |
| total| 160.418988 s|  2.201932 s | 1.267406 s |


```text
$ nvaccelinfo 

CUDA Driver Version:           12020
NVRM version:                  NVIDIA UNIX x86_64 Kernel Module  535.230.02  Fri Dec 20 21:42:05 UTC 2024

Device Number:                 0
Device Name:                   NVIDIA GeForce RTX 4070 Ti
Device Revision Number:        8.9
Global Memory Size:            12590645248
Number of Multiprocessors:     60
Concurrent Copy and Execution: Yes
Total Constant Memory:         65536
Total Shared Memory per Block: 49152
Registers per Block:           65536
Warp Size:                     32
Maximum Threads per Block:     1024
Maximum Block Dimensions:      1024, 1024, 64
Maximum Grid Dimensions:       2147483647 x 65535 x 65535
Maximum Memory Pitch:          2147483647B
Texture Alignment:             512B
Clock Rate:                    2625 MHz
Execution Timeout:             No
Integrated Device:             No
Can Map Host Memory:           Yes
Compute Mode:                  default
Concurrent Kernels:            Yes
ECC Enabled:                   No
Memory Clock Rate:             10501 MHz
Memory Bus Width:              192 bits
L2 Cache Size:                 50331648 bytes
Max Threads Per SMP:           1536
Async Engines:                 2
Unified Addressing:            Yes
Managed Memory:                Yes
Concurrent Managed Memory:     Yes
Preemption Supported:          Yes
Cooperative Launch:            Yes
Unified Memory:                No
Memory Models Flags:           -gpu=mem:separate, -gpu=mem:managed
Default Target:                cc89
```