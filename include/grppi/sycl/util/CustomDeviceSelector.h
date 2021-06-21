//
// Created by veigas on 20/6/21.
//

#ifndef GRPPI_CUSTOMDEVICESELECTOR_H
#define GRPPI_CUSTOMDEVICESELECTOR_H

#include <CL/sycl.hpp>

namespace grppi::sycl_utils {
    class CustomDeviceSelector : public sycl::device_selector {

    public:

        CustomDeviceSelector(unsigned platform, unsigned device) : platform{platform}, device{device} {}

        int operator()(const cl::sycl::device &device_iter) const override {
          if (device_iter.get_platform().get_platforms()[platform].get_devices()[device] == device_iter) return 100;
          return -1;
        }

    private:
        unsigned platform;
        unsigned device;

    };
}


#endif //GRPPI_CUSTOMDEVICESELECTOR_H