# SYCL backend prerequisites

In order to use the SYCL backend, the following components are required:
* SYCL compiler.
* OpenCL headers.
* OpenCL ICD loader.
* Device drivers & ICDs.

## Installation

### SYCL compilers
**IMPORTANT:** As of the time of writing, the SYCL backend is using aspects of the SYCL 2020 standard which might not be fully
implemented by certain compilers.

The following compilers have been tested for GrPPI:
* [ComputeCpp&trade; Community Edition](https://developer.codeplay.com/products/computecpp/ce/home/).
* [Intel OneAPI DPC++ Goldmaster](https://software.intel.com/content/www/us/en/develop/tools/oneapi/data-parallel-c-plus-plus.html).
* [Intel LLVM DPC++](https://github.com/intel/llvm/tree/sycl/sycl).

The CMake project supports additional custom SYCL compilers without warranties. However, this should only impact the building process of
sample code and units tests, not the installation or use of GrPPI interface. 

### OpenCL development files
It's possible to obtain the necessary OpenCL files by running the following commands:

#### Ubuntu / Debian

```shell
sudo apt install opencl-headers      # OpenCL Headers
sudo apt install ocl-icd-opencl-dev  # OpenCL ICD Loader
```
An alternative ICD Loader is Khronos' [OpenCL ICD Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader)

### Device drivers
SYCL supports a wide range of devices such as CPUs, GPUs, FPGAs, and more. However OpenCL-capable drivers and drivers are required.

**IMPORTANT:** SPIR and SPIR-V are the common supported bitcodes. Support for AMD and Nvidia GPUs is limited as they use AMDGCN and PTX device code respectively.
Currently DPC++ has experimental support for Nvidia GPUs but you must build the compiler from the source repository as it's not included in the OneAPI Toolkit; support for AMD GPUs is planned for the future.
ComputeCpp also experimentally supports Nvidia GPUs, however they recently marked the capability as unsupported. 

#### Intel devices
You can find the necessary drivers for Intel devices [here](https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html).

#### Nvidia devices
```shell
sudo apt install nvidia-driver-<VERSION> # Devices drivers
sudo apt install nvidia-opencl-dev       # OpenCL files
```

#### AMD devices
There are two frameworks that can be installed for the use of AMD devices, however both should not be present at the same time.
* [Radeon&trade; Software for Linux (amdgpupro)](https://www.amd.com/en/support/kb/release-notes/rn-amdgpu-unified-linux-20-50)
  \# Version 20.50 dropped support for PAL OpenCL for ROCr instead.
* [Radeon&trade; Open Compute (ROCm)](https://github.com/RadeonOpenCompute/ROCm)

**NOTE**: Please be mindful of the required kernel versions for these drivers. For Ubuntu 20.04 it will **only** build on version 5.4.0.
Support for GPUs is currently expected for ROCm drivers. 