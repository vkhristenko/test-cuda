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
 - It might complain - ignore
 ```
 WARNING: nvidia-installer was forced to guess the X library path '/usr/lib64' and X module path '/usr/lib64/xorg/modules'; these paths were not queryable from the system.  If X fails to find the NVIDIA X driver module, please install the `pkg-config` utility and the X.Org SDK/development package for your distribution and reinstall the driver.
 ```
