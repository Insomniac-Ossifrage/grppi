/*
 * Copyright 2021 Universidad Carlos III de Madrid
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GRPPI_SYCL_PARALLEL_EXECUTION_H
#define GRPPI_SYCL_PARALLEL_EXECUTION_H

#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/callable_traits.h"
#include "../common/execution_traits.h"
#include "../common/patterns.h"
#include "../common/pack_traits.h"

#include <type_traits>
#include <tuple>
#include <iterator>
#include <functional>
#include "kernels/sycl_kernels.h"

namespace grppi {
/**
\brief SYCL Parallel execution policy.
*/
class parallel_execution_sycl {

public:

  /// \brief Default constructor.
  parallel_execution_sycl(): queue_{sycl::cpu_selector{}, [](const sycl::exception_list &exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        std::cout << e.what() << std::endl;
      }
    }}} {};

  /**
  \brief Applies a transformation to multiple sequences leaving the result in
  another sequence.
  \tparam InputIterators Iterator types for input sequences.
  \tparam OutputIterator Iterator type for the output sequence.
  \tparam Transformer Callable object type for the transformation.
  \param firsts Tuple of iterators to input sequences.
  \param first_out Iterator to the output sequence.
  \param sequence_size Size of the input sequences.
  \param transform_op Transformation callable object.
  \pre For every I iterators in the range 
       `[get<I>(firsts), next(get<I>(firsts),sequence_size))` are valid.
  \pre Iterators in the range `[first_out, next(first_out,sequence_size)]` are valid.
  */
  template <typename ... InputIterators, typename OutputIterator, 
            typename Transformer>
  void map(std::tuple<InputIterators...> firsts,
      OutputIterator first_out, std::size_t sequence_size, 
      Transformer && transform_op) const;
  
  /**
  \brief Applies a reduction to a sequence of data items. 
  \tparam InputIterator Iterator type for the input sequence.
  \tparam Identity Type for the identity value.
  \tparam Combiner Callable object type for the combination.
  \param first Iterator to the first element of the sequence.
  \param sequence_size Size of the input sequence.
  \param last Iterator to one past the end of the sequence.
  \param identity Identity value for the reduction.
  \param combine_op Combination callable object.
  \pre Iterators in the range `[first,last)` are valid. 
  \return The reduction result
  */
  template <typename InputIterator, typename Identity, typename Combiner>
  auto reduce(InputIterator first, std::size_t sequence_size,
              Identity && identity,
              Combiner && combine_op) const;

  /**
  \brief Applies a map/reduce operation to a sequence of data items.
  \tparam InputIterator Iterator type for the input sequence.
  \tparam Identity Type for the identity value.
  \tparam Transformer Callable object type for the transformation.
  \tparam Combiner Callable object type for the combination.
  \param first Iterator to the first element of the sequence.
  \param sequence_size Size of the input sequence.
  \param identity Identity value for the reduction.
  \param transform_op Transformation callable object.
  \param combine_op Combination callable object.
  \pre Iterators in the range `[first,last)` are valid. 
  \return The map/reduce result.
  */
  template <typename ... InputIterators, typename Identity, 
            typename Transformer, typename Combiner>
  auto map_reduce(std::tuple<InputIterators...> firsts,
                  std::size_t sequence_size,
                  Identity && identity,
                  Transformer && transform_op, Combiner && combine_op) const;

  /**
  \brief Applies a stencil to multiple sequences leaving the result in
  another sequence.
  \tparam InputIterators Iterator types for input sequences.
  \tparam OutputIterator Iterator type for the output sequence.
  \tparam StencilTransformer Callable object type for the stencil transformation.
  \tparam Neighbourhood Callable object for generating neighbourhoods.
  \param firsts Tuple of iterators to input sequences.
  \param first_out Iterator to the output sequence.
  \param sequence_size Size of the input sequences.
  \param transform_op Stencil transformation callable object.
  \param neighbour_op Neighbourhood callable object.
  \pre For every I iterators in the range 
       `[get<I>(firsts), next(get<I>(firsts),sequence_size))` are valid.
  \pre Iterators in the range `[first_out, next(first_out,sequence_size)]` are valid.
  */
  template <typename ... InputIterators, typename OutputIterator,
            typename StencilTransformer, typename Neighbourhood>
  constexpr void stencil(std::tuple<InputIterators...> firsts, OutputIterator first_out,
               std::size_t sequence_size,
               StencilTransformer && transform_op,
               Neighbourhood && neighbour_op) const;

  /**
  \brief Invoke \ref md_divide-conquer.
  \tparam Input Type used for the input problem.
  \tparam Divider Callable type for the divider operation.
  \tparam Predicate Callable type for the stop condition predicate.
  \tparam Solver Callable type for the solver operation.
  \tparam Combiner Callable type for the combiner operation.
  \param ex Sequential execution policy object.
  \param input Input problem to be solved.
  \param divider_op Divider operation.
  \param predicate_op Predicate operation.
  \param solver_op Solver operation.
  \param combine_op Combiner operation.
  */
  template <typename Input, typename Divider,typename Predicate, typename Solver, typename Combiner>
  auto divide_conquer(Input && input,
                      Divider && divide_op,
                      Predicate && predicate_op,
                      Solver && solve_op,
                      Combiner && combine_op) const;

  /**
  \brief Invoke \ref md_pipeline.
  \tparam Generator Callable type for the generator operation.
  \tparam Transformers Callable types for the transformers in the pipeline.
  \param generate_op Generator operation.
  \param transform_ops Transformer operations.
  */
  template <typename Generator, typename ... Transformers>
  void pipeline(Generator && generate_op,
                Transformers && ... transform_op) const;

private:
    sycl::queue queue_;
    size_t work_group_load_ = 256;

};

/// Determine if a type is a parallel execution policy.
template <typename E>
constexpr bool is_parallel_execution_sycl() {
  return true;
}

/**
\brief Determines if an execution policy is supported in the current compilation.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool is_supported<parallel_execution_sycl>() { return true; }

/**
\brief Determines if an execution policy supports the map pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_map<parallel_execution_sycl>() { return true; }

/**
\brief Determines if an execution policy supports the reduce pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_reduce<parallel_execution_sycl>() { return true; }

/**
\brief Determines if an execution policy supports the map-reduce pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_map_reduce<parallel_execution_sycl>() { return true; }

/**
\brief Determines if an execution policy supports the stencil pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_stencil<parallel_execution_sycl>() { return false; }

/**
\brief Determines if an execution policy supports the divide/conquer pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_divide_conquer<parallel_execution_sycl>() { return false; }

/**
\brief Determines if an execution policy supports the pipeline pattern.
\note Specialization for sequential_execution.
*/
template <>
constexpr bool supports_pipeline<parallel_execution_sycl>() { return false; }

template <typename ... InputIterators, typename OutputIterator,
          typename Transformer>
void parallel_execution_sycl::map(
    std::tuple<InputIterators...> firsts,
    OutputIterator first_out, 
    std::size_t sequence_size, 
    Transformer && transform_op) const
{
  // Safeguard
  if (sequence_size < 1) return;
  // Types
  using Input_T = typename std::iterator_traits<std::tuple_element_t<0, std::tuple<InputIterators...>>>::value_type;
  using Output_T = typename std::iterator_traits<OutputIterator>::value_type;
  // Input Iterators
  std::array in_buffers = {std::apply([sequence_size](const auto&... inputs){
    std::array collection{sycl::buffer<Input_T,1>{inputs, inputs + sequence_size}...};
    return collection;
    }, firsts)};
  // Output Iterator
  sycl::buffer<Output_T , 1> out_buffer{first_out, first_out + sequence_size};
  // Kernel Call
  sycl_kernel::template map<Output_T, in_buffers.size()>(queue_, sequence_size, in_buffers, out_buffer, std::forward<Transformer>(transform_op));
  // Write back to OutputIterator
  out_buffer.template set_final_data(first_out);
}

#ifdef GRPPI_SYCL_EXPERIMENTAL_REDUCTION
template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_sycl::reduce(
    InputIterator first,
    std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  // Output Value
  using T = typename std::iterator_traits<InputIterator>::value_type;
  T result = identity;
  {
    // Buffers
    sycl::buffer<T, 1> res_buffer{&result, 1};
    sycl::buffer<T, 1> in_buffer{first, first + sequence_size};
    // Reduction Parameters
    // TODO Review ND-Range parameters for better parallelism
    unsigned long max_workgroup_size{queue_.get_device().template get_info<sycl::info::device::max_work_group_size>()/12};
    if (max_workgroup_size == 0) max_workgroup_size++;
    unsigned long remaining_workitems{sequence_size};
    unsigned long offset{0};
    do {
      unsigned long current_worksgroup_size = std::min(max_workgroup_size, remaining_workitems);
      remaining_workitems -= current_worksgroup_size;
      // Queue
      const_cast<sycl::queue &>(queue_).template submit([&](sycl::handler &cgh) {
          auto in_acc = in_buffer.template get_access<sycl::access::mode::read>(cgh, sycl::range<1>{current_worksgroup_size}, sycl::id<1>{offset});
          auto op_reduction = sycl::reduction(res_buffer, cgh, identity, combine_op);
          // TODO Kernel Name
          cgh.template parallel_for<class ReductionKernel>(sycl::nd_range<1>{current_worksgroup_size, current_worksgroup_size}, op_reduction,
                  [=](sycl::nd_item<1> idx, auto &op) {
                      op.combine(in_acc[idx.get_global_linear_id()]);
                  });
      });
      // TODO Error Handling
      const_cast<sycl::queue &>(queue_).wait();
      offset += current_worksgroup_size;
    } while (remaining_workitems > 0);
  } // Wait for buffer destruction.

  return result;
}
#else
template <typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_sycl::reduce(
    InputIterator first,
    std::size_t sequence_size,
    Identity && identity,
    Combiner && combine_op) const
{
  // Safeguard
  if (sequence_size < 1) return identity;
  else if (sequence_size == 1) return combine_op(*first, *first);
  // Output Value
  using T = typename std::iterator_traits<InputIterator>::value_type;
  T result = identity;
  {
    // Buffers
    sycl::buffer<T, 1> in_buffer{first, first + sequence_size};
    sycl::buffer<T, 1> out_buffer{&result, sycl::range<1>(1)};
    in_buffer.template set_final_data(nullptr);
    // Interface
    grppi::sycl_kernel::reduce(queue_, sequence_size, in_buffer, out_buffer, std::forward<Identity>(identity), std::forward<Combiner>(combine_op));

  }
  return result;
}
#endif

template <typename ... InputIterators, typename Identity, 
          typename Transformer, typename Combiner>
auto parallel_execution_sycl::map_reduce(
    std::tuple<InputIterators...> firsts,
    std::size_t sequence_size, 
    Identity && identity,
    Transformer && transform_op, Combiner && combine_op) const
{
  // Safeguard
  if (sequence_size < 1) return identity;
  else if (sequence_size == 1) return combine_op(identity, apply_deref_increment(std::forward<Transformer>(transform_op), firsts));
  // Input iterators to SYCL buffers
  using Input_T = typename std::iterator_traits<std::tuple_element_t<0, std::tuple<InputIterators...>>>::value_type; // TODO: Simplify
  std::array in_buffers = {std::apply([sequence_size](const auto&... inputs){
      std::array collection{sycl::buffer<Input_T,1>{inputs, inputs + sequence_size}...};
      return collection;
  }, firsts)};

  return grppi::sycl_kernel::map_reduce<Input_T>(queue_, sequence_size, in_buffers, std::forward<Transformer>(transform_op), std::forward<Identity>(identity), std::forward<Combiner>(combine_op));
}

template <typename ... InputIterators, typename OutputIterator,
          typename StencilTransformer, typename Neighbourhood>
constexpr void parallel_execution_sycl::stencil(
    std::tuple<InputIterators...> firsts, OutputIterator first_out,
    std::size_t sequence_size,
    StencilTransformer && transform_op,
    Neighbourhood && neighbour_op) const
{
}

template <typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_sycl::divide_conquer(
    Input && input,
    Divider && divide_op,
    Predicate && predicate_op,
    Solver && solve_op,
    Combiner && combine_op) const
{
  //TODO Remove Placeholder
  if (predicate_op(input)) { return solve_op(std::forward<Input>(input)); }
  auto subproblems = divide_op(std::forward<Input>(input));

  using subproblem_type =
  std::decay_t<typename std::result_of<Solver(Input)>::type>;
  std::vector<subproblem_type> solutions;
  for (auto && sp : subproblems) {
    solutions.push_back(divide_conquer(sp,
                                       std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),std::forward<Solver>(solve_op),
                                       std::forward<Combiner>(combine_op)));
  }
  return reduce(std::next(solutions.begin()), solutions.size()-1, solutions[0],
                std::forward<Combiner>(combine_op));
}

template <typename Generator, typename ... Transformers>
void parallel_execution_sycl::pipeline(
        Generator && generate_op,
        Transformers && ... transform_ops) const
{
}

} // end namespace grppi

#endif
