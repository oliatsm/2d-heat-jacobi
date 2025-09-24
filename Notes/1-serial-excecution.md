# Serial Excecution

```
$ ./laplace 
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
 total: 108.779522 s
 ```

 ## CPU info

 ```
 Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          48 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   16
  On-line CPU(s) list:    0-15
Vendor ID:                AuthenticAMD
  Model name:             AMD Ryzen 7 5800X 8-Core Processor
    CPU family:           25
    Model:                33
    Thread(s) per core:   2
    Core(s) per socket:   8
    Socket(s):            1
    Stepping:             2
    Frequency boost:      enabled
    CPU max MHz:          4850,1948
    CPU min MHz:          2200,0000
    BogoMIPS:             7586.38
```

# Serial Execution Asus

```
$ ./laplace 
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
 total: 267.478489 s
```

## lscpu

```
$ lscpu
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:                           Little Endian
Address sizes:                        39 bits physical, 48 bits virtual
CPU(s):                               4
On-line CPU(s) list:                  0-3
Thread(s) per core:                   2
Core(s) per socket:                   2
Socket(s):                            1
NUMA node(s):                         1
Vendor ID:                            GenuineIntel
CPU family:                           6
Model:                                61
Model name:                           Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
Stepping:                             4
CPU MHz:                              2215.837
CPU max MHz:                          2700,0000
CPU min MHz:                          500,0000
BogoMIPS:                             4393.67
Virtualization:                       VT-x
L1d cache:                            64 KiB
L1i cache:                            64 KiB
L2 cache:                             512 KiB
L3 cache:                             3 MiB
```
