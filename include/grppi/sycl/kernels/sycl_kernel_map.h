#ifndef GRPPI_SYCL_KERNEL_MAP_H
#define GRPPI_SYCL_KERNEL_MAP_H

#include <iostream>
#include <CL/sycl.hpp>

namespace grppi::sycl_kernel {

template<typename input_accessor_t, typename output_accessor_t, typename Transformer>
class MapKernelFunctor {
private:
  const Transformer transform_op_;
  const input_accessor_t input_;
  output_accessor_t output_;
public:
  MapKernelFunctor(
    input_accessor_t input,
    output_accessor_t output,
    Transformer transformer_op)
    :
    input_{input},
    output_{output},
    transform_op_{std::move(transformer_op)} {}

  void inline operator() (sycl::id<1> id) const {
    output_[id] = std::apply([&](const auto &...accessors){
        return transform_op_(accessors[id]...);
    }, input_);
  }
};

/**
 * Main interface used to call the different kernels for the map pattern. Currently there is only one basic kernel.
 * @tparam data_t Type of the used data.
 * @tparam array_size Number of input containers.
 * @tparam Transformer Typename used to define a lambda expression.
 * @param queue SYCL queue to submit the kernel to.
 * @param sequence_size Size of the input containers.
 * @param input_buffers std::array composed of SYCL buffers mapped to the input iterators.
 * @param output_buffer SYCL buffer mapped to the output iterator.
 * @param transform_op Operation to be performed to the input data.
 */
template<typename data_t, size_t array_size, typename Transformer>
inline void map(
  const sycl::queue &queue,
  const size_t sequence_size,
  std::array<sycl::buffer<data_t, 1>, array_size> &input_buffers,
  sycl::buffer<data_t, 1> &output_buffer,
  Transformer &&transform_op
  )
{
  // Queue
  const_cast<sycl::queue &>(queue).template submit([&](sycl::handler &cgh) {
    // Input Accessors
    std::array in_accs = {std::apply([&] (auto&... buffers) {
        std::array accessors{buffers.template get_access<sycl::access::mode::read>(cgh)...};
        return accessors;
    },input_buffers)};

    // Output Accessor
    auto out_acc{output_buffer.template get_access<sycl::access::mode::write>(cgh)};

    cgh.template parallel_for(sycl::range<1>{sequence_size},
      MapKernelFunctor{in_accs, out_acc, std::forward<Transformer>(transform_op)});
  });
  // TODO: Handle Exceptions
  try {
    const_cast<sycl::queue &>(queue).wait_and_throw();
  } catch (sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }

}

} // end namespace grppi::sycl_kernel

#endif //GRPPI_SYCL_KERNEL_MAP_H
