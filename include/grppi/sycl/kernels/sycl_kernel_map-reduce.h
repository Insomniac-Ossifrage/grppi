#ifndef GRPPI_SYCL_KERNEL_MAP_REDUCE_H
#define GRPPI_SYCL_KERNEL_MAP_REDUCE_H

namespace grppi::sycl_kernel {


template<typename Input>
class MapReduceFunctor {

};

template <typename data_t, typename Input, typename Transformer, typename Identity, typename Combiner>
inline auto map_reduce(
  const sycl::queue &queue,
  const size_t sequence_size,
  const Input &input_array

  ) {

}


} // grppi::sycl_kernel

#endif //GRPPI_SYCL_KERNEL_MAP_REDUCE_H
