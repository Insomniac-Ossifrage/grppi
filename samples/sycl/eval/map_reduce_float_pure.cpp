#include <iostream>
#include <vector>
#include "grppi/dyn/dynamic_execution.h"



int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Invalid number of arguments." << "\n";
    return -1;
  }

  unsigned sequence_size(std::stol(argv[4]));
  std::vector<float> vector1{};
  std::vector<float> vector2{};
  // Initialize
  for (long i{0}; i < sequence_size; ++i) {
    vector1.push_back(static_cast<float>(i)/sequence_size);
    vector2.push_back(static_cast<float>(i));
  }
  float result = 0.0F;
  auto ex = grppi::parallel_execution_sycl(std::stoi(argv[2]), std::stoi(argv[3]));
  auto t1 = std::chrono::system_clock::now();
  auto b_vector1 = sycl::buffer(vector1.data(), sycl::range<1>(vector1.size()));
  auto b_vector2 = sycl::buffer(vector2.data(), sycl::range<1>(vector2.size()));
  auto b_result = sycl::buffer{&result, sycl::range<1>(1)};
  std::array<sycl::buffer<float, 1>, 2> b_input {b_vector1, b_vector2};
  grppi::sycl_kernel::map_reduce<float>(ex.get_queue(), vector1.size(), b_input,[] (auto a, auto b)->float {return a*b;}, 1.0F, [](auto a, auto b)->float {return a-b;});
  auto t2 = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
  std::cout << sequence_size  << ", " << diff.count() << std::endl;

  return 0;
}