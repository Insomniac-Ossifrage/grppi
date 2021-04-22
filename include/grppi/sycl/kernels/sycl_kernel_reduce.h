#ifndef GRPPI_SYCL_KERNEL_REDUCE_H
#define GRPPI_SYCL_KERNEL_REDUCE_H

namespace grppi::sycl_kernel {

template<typename input_accessor, typename temp_accessor, typename local_accessor, typename Identity, typename Combiner>
class ReduceKernelFunctor{

};

template<typename data_t, typename Identity, typename Combiner, size_t work_group_load=256>
inline void reduce(
  const sycl::queue &queue,
  const size_t sequence_size,
  sycl::buffer<data_t, 1> &input_buffer,
  sycl::buffer<data_t, 1> &output_buffer,
  Identity &&identity,
  Combiner &&combine_op
  ) {
  // Parameters
  const constexpr size_t k_factor = 2;
  // Data
  const size_t local_size = work_group_load / k_factor;
  const size_t global_size = (((sequence_size/k_factor) + local_size - 1) / local_size) * local_size;
  size_t num_workgroups = global_size / local_size;
  // Conditional buffer
  auto temp_buffer = (num_workgroups > 1) ? sycl::buffer<data_t,1>{sycl::range<1>(num_workgroups)} : output_buffer;
  // Queue
  const_cast<sycl::queue &>(queue).template submit([&](sycl::handler &cgh) {
    // Accessors
    auto in_acc = input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto temp_acc = temp_buffer.template get_access<sycl::access::mode::write>(cgh);
    sycl::accessor<data_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_acc{sycl::range<1>{local_size}, cgh};
    // Range
    cgh.template parallel_for<class Red_Kernel>(sycl::nd_range<1>{global_size, local_size}, [=] (sycl::nd_item<1> item) {
      // Indexes
      size_t global_id = item.get_global_id(0);
      size_t local_id = item.get_local_id(0);
      Identity private_memory = identity;
      // Thread Reduction
      // Global range can be < sequence_size. This reduction is optimal when global_range = sequence_size/2
      for (size_t i = global_id; i < sequence_size; i+= item.get_global_range(0)) {
        private_memory = combine_op(private_memory, ((i < sequence_size) ? in_acc[i] : identity));
      }
      local_acc[local_id] = private_memory;
      // Stride
      // The input accessor must have even elements or the last element won't be reduced
      for (size_t i = local_size / 2; i > 0; i >>= 1) {
        // Local barrier for items in work group
        // TODO Update to SYCL 2020 function as this one is deprecated.
        item.barrier(sycl::access::fence_space::local_space);
        if (local_id < i) local_acc[local_id] = combine_op(local_acc[local_id], local_acc[local_id + i]);
      }
      // Saving result
      if (local_id == 0) temp_acc[item.get_group(0)] = local_acc[0];
    });
  });
  try {
    const_cast<sycl::queue &>(queue).wait_and_throw();
  } catch (sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }

  // Remaining reduction
  // TODO Implement parallel reduction for larger cases
  if (num_workgroups > 1) {
    auto host_out_acc = output_buffer.template get_access<cl::sycl::access::mode::write>();
    auto host_in_acc = temp_buffer.template get_access<cl::sycl::access::mode::read>();
    // Host reduction
    for (size_t i = 0; i < num_workgroups; i++) {
      host_out_acc[0] += host_in_acc[i];
    }
  }
}

} // end namespace grppi::sycl_kernel



#endif //GRPPI_SYCL_KERNEL_REDUCE_H
