#ifndef GRPPI_SYCL_KERNEL_MAP_REDUCE_H
#define GRPPI_SYCL_KERNEL_MAP_REDUCE_H

namespace grppi::sycl_kernel {


template<typename Input_Accessor, typename Temp_Accessor, typename Local_Accessor, typename Transformer, typename Identity, typename Combiner>
class MapReduceFunctor {
private:
  Input_Accessor input_;
  Temp_Accessor temp_;
  Local_Accessor local_;
  const size_t sequence_size_;
  const size_t local_size_;
  const Transformer transformer_;
  const Identity identity_;
  const Combiner combine_op_;

public:
  MapReduceFunctor(Input_Accessor &input_acc,
                   Temp_Accessor &temp_acc,
                   Local_Accessor &local_acc,
                   size_t sequence_size,
                   size_t local_size,
                   Transformer transformer,
                   Identity identity,
                   Combiner combiner)
                   :
                   input_{input_acc},
                   temp_{temp_acc},
                   local_{local_acc},
                   sequence_size_{sequence_size},
                   local_size_{local_size},
                   transformer_{transformer},
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
      private_memory = combine_op_(private_memory, ((i < sequence_size_) ?
      std::apply([&](const auto &...accessors){return transformer_(accessors[i]...);}, input_)
      : identity_));
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

template <typename data_t, typename Input, typename Transformer, typename Identity, typename Combiner, size_t work_group_load = 256>
inline auto map_reduce(
  const sycl::queue &queue,
  const size_t sequence_size,
  Input &input_array,
  Transformer &&transformer,
  Identity &&identity,
  Combiner &&combiner
  ) {
  std::cout << typeid(data_t).name() <<'\n';
  // Parameters
  const constexpr size_t k_factor = 2;
  // Data
  const size_t local_size = work_group_load / k_factor;
  const size_t global_size = (((sequence_size/k_factor) + local_size - 1) / local_size) * local_size;
  size_t num_workgroups = global_size / local_size;
  // Conditional buffer
  auto temp_buffer = (num_workgroups > 1) ? sycl::buffer<data_t,1>{sycl::range<1>(num_workgroups)} : sycl::buffer<data_t, 1>{sycl::range<1>{1}};
  // Queue
  const_cast<sycl::queue &>(queue).template submit([&](sycl::handler &cgh) {
      // Accessors
      std::array in_accs = {std::apply([&] (auto&... buffers) {
          std::array accessors{buffers.template get_access<sycl::access::mode::read>(cgh)...};
          return accessors;
      },input_array)};
      auto temp_acc = temp_buffer.template get_access<sycl::access::mode::write>(cgh);
      sycl::accessor<data_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_acc{sycl::range<1>{local_size}, cgh};
      // Launching Kernel
      cgh.template parallel_for(sycl::nd_range<1>{global_size, local_size},
        MapReduceFunctor{in_accs, temp_acc, local_acc, sequence_size, local_size, transformer, identity, combiner});
  });
  try {
    const_cast<sycl::queue &>(queue).wait_and_throw();
  } catch (sycl::exception const& e) {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
  // Obtain and return the result of the map-reduction
  {
    // TODO Implement parallel reduction for larger cases
    auto host_in_acc = temp_buffer.template get_access<cl::sycl::access::mode::read_write>();
    if (num_workgroups > 1) {
      // Host reduction
      for (size_t i = 0; i < num_workgroups; i++) {
        host_in_acc[0] = combiner(host_in_acc[0], host_in_acc[i]);
      }
    }
    return host_in_acc[0];
  }
}

} // grppi::sycl_kernel

#endif //GRPPI_SYCL_KERNEL_MAP_REDUCE_H
