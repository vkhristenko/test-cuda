# Installation Guide for CUDA + Docker Tests

Start from scratch, Centos 7 VM. No cuda drivers

## create a user with sudo rights
- add a new user: `adduser cmscuda`
- set up the password: `passwd cmscuda`
- add _cmscuda_ to the _wheel_ group: `usermod -aG wheel cmscuda`
- test: `su - cmscuda`

## Installing Nvidia Drivers

### 1. Find out the type of card
First, need to know which card we have
```
[root@bench-dev-gpu cmscuda]# lshw -class display
  *-display:0
       description: VGA compatible controller
       product: GD 5446
       vendor: Cirrus Logic
       physical id: 2
       bus info: pci@0000:00:02.0
       version: 00
       width: 32 bits
       clock: 33MHz
       capabilities: vga_controller rom
       configuration: driver=cirrus latency=0
       resources: irq:0 memory:fa000000-fbffffff memory:fe050000-fe050fff memory:fe040000-fe04ffff
  *-display:1
       description: 3D controller
       product: GV100GL [Tesla V100 PCIe 32GB]
       vendor: NVIDIA Corporation
       physical id: 5
       bus info: pci@0000:00:05.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: pm msi pciexpress bus_master cap_list
       configuration: driver=nouveau latency=0
       resources: iomemory:80-7f iomemory:100-ff irq:29 memory:fd000000-fdffffff memory:800000000-fffffffff memory:1000000000-1001ffffff
```

_We have Tesla Architecture V100 card_

### 2. Get the driver
- go to [Nvidia](https://www.nvidia.com/Download/index.aspx?lang=en-us) and select accordingly

### 3. Follow something
- [Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) goes directly to install _CUDA_ distribution. 
- I followed [this](https://www.advancedclustering.com/act_kb/installing-nvidia-drivers-rhel-centos-7/) to install just drivers. worked 100%. 
 - It might complain as below - _ignore_
 ```
 WARNING: nvidia-installer was forced to guess the X library path '/usr/lib64' and X module path '/usr/lib64/xorg/modules'; 
 these paths were not queryable from the system.  
 If X fails to find the NVIDIA X driver module, please install the `pkg-config` 
 utility and the X.Org SDK/development package for your distribution and reinstall the driver.
 ```

### 4. Test 
```
[root@bench-dev-gpu cmscuda]# nvidia-smi
Mon May 20 18:06:58 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.14       Driver Version: 430.14       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:00:05.0 Off |                    0 |
| N/A   41C    P0    36W / 250W |      0MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
[root@bench-dev-gpu cmscuda]#
```

## Install CUDA Toolkit
- In principle continue following this [link](https://www.advancedclustering.com/act_kb/installing-nvidia-drivers-rhel-centos-7/)
- install _CUDA Toolkit_, excluding driver installation.
 - note, that it is also possible to not install drivers separately... but to pull them with cuda toolkit

## Testing CUDA Samples
- samples are in `/usr/local/cuda/samples`
- _Already compiled_
- Run _deviceQuery_:
```
[root@bench-dev-gpu samples]# ./1_Utilities/deviceQuery/deviceQuery
./1_Utilities/deviceQuery/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Tesla V100-PCIE-32GB"
  CUDA Driver Version / Runtime Version          10.2 / 10.1
  CUDA Capability Major/Minor version number:    7.0
  Total amount of global memory:                 32510 MBytes (34089730048 bytes)
  (80) Multiprocessors, ( 64) CUDA Cores/MP:     5120 CUDA Cores
  GPU Max Clock rate:                            1380 MHz (1.38 GHz)
  Memory Clock rate:                             877 Mhz
  Memory Bus Width:                              4096-bit
  L2 Cache Size:                                 6291456 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 7 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 5
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.2, CUDA Runtime Version = 10.1, NumDevs = 1
Result = PASS
[root@bench-dev-gpu samples]#
```
- Run Vector Add
```
[root@bench-dev-gpu samples]# ./0_Simple/vectorAdd/vectorAdd
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
[root@bench-dev-gpu samples]#
```

- Run Matrix Mult
```
[root@bench-dev-gpu samples]# ./0_Simple/matrixMul/matrixMul
[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "Tesla V100-PCIE-32GB" with compute capability 7.0

MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 2633.20 GFlop/s, Time= 0.050 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS

NOTE: The CUDA Samples are not meant for performancemeasurements. Results may vary when GPU Boost is enabled.
[root@bench-dev-gpu samples]#
```
