#ifndef GRPPI_SYCL_KERNEL_REDUCE_H
#define GRPPI_SYCL_KERNEL_REDUCE_H

namespace grppi::sycl_kernel {

template<typename Input_Accessor, typename Temp_Accessor, typename Local_Accessor, typename Identity, typename Combiner>
class ReduceKernelFunctor{
private:
    Input_Accessor input_;
    Temp_Accessor temp_;
    Local_Accessor local_;
    const size_t sequence_size_;
    const size_t local_size_;
    const Identity identity_;
    const Combiner combine_op_;
public:
    ReduceKernelFunctor(Input_Accessor &input_acc,
                        Temp_Accessor &temp_acc,
                        Local_Accessor &local_acc,
                        size_t sequence_size,
                        size_t local_size,
                        Identity identity,
                        Combiner combiner)
                        :
                        input_{input_acc},
                        temp_{temp_acc},
                        local_{local_acc},
                        sequence_size_{sequence_size},
                        local_size_{local_size},
                        identity_{identity},
                        combine_op_{combiner}
                        {};

    void inline operator() (sycl::nd_item<1> item) const {
      // Indexes
      size_t global_id = item.get_global_id(0);
      size_t local_id = item.get_local_id(0);
      Identity private_memory = identity_;
      // Thread Reduction
      // Global range can be < sequence_size. This reduction is optimal when global_range = sequence_size/2
      for (size_t i = global_id; i < sequence_size_; i+= item.get_global_range(0)) {
        private_memory = combine_op_(private_memory, ((i < sequence_size_) ? input_[i] : identity_));
      }
      local_[local_id] = private_memory;
      // Stride
      // The input accessor must have even elements or the last element won't be reduced
      for (size_t i = local_size_ / 2; i > 0; i >>= 1) {
        // Local barrier for items in work group
        // TODO Update to SYCL 2020 function as this one is deprecated.
        item.barrier(sycl::access::fence_space::local_space);
        if (local_id < i) local_[local_id] = combine_op_(local_[local_id], local_[local_id + i]);
      }
      // Saving result
      if (local_id == 0) temp_[item.get_group(0)] = local_[0];
    }
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
  // Data
  const size_t local_size = work_group_load / 2;
  const size_t global_size = (((sequence_size/2) + local_size - 1) / local_size) * local_size;
  size_t num_workgroups = global_size / local_size;
  // Conditional buffer
  auto temp_buffer = (num_workgroups > 1) ? sycl::buffer<data_t,1>{sycl::range<1>(num_workgroups)} : output_buffer;
  // Queue
  const_cast<sycl::queue &>(queue).template submit([&](sycl::handler &cgh) {
    // Accessors
    auto in_acc = input_buffer.template get_access<sycl::access::mode::read>(cgh);
    auto temp_acc = temp_buffer.template get_access<sycl::access::mode::write>(cgh);
    sycl::accessor<data_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_acc{sycl::range<1>{local_size}, cgh};
    // Launching Kernel
    cgh.template parallel_for(sycl::nd_range<1>{global_size, local_size},
      ReduceKernelFunctor{in_acc, temp_acc, local_acc, sequence_size, local_size, identity, combine_op});
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
      host_out_acc[0] = combine_op(host_out_acc[0], host_in_acc[i]);
    }
  }
}

} // end namespace grppi::sycl_kernel



#endif //GRPPI_SYCL_KERNEL_REDUCE_H
