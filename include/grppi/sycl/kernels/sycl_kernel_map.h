#ifndef GRPPI_SYCL_KERNEL_MAP_H
#define GRPPI_SYCL_KERNEL_MAP_H

#import <CL/sycl.hpp>

namespace grppi::sycl::kernels {
template <typename InputAccessors...>
class SYCL_Kernel_Map {

    void operator() (cl::sycl::item<1> item) {

    }
};

}

#endif //GRPPI_SYCL_KERNEL_MAP_H
