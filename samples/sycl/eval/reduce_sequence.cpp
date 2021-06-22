#include <iostream>
#include <complex>
#include <vector>
#include "grppi/grppi.h"
#include "grppi/dyn/dynamic_execution.h"


grppi::dynamic_execution select_backend(const std::string &mode, unsigned platform, unsigned device) {
  if ("sycl" == mode) return grppi::parallel_execution_sycl{platform, device};
  if ("nat" == mode) return  grppi::parallel_execution_native{};
  return {};
}

void direct(char **argv) {
  unsigned sequence_size(std::stol(argv[4]));
  std::vector<long> data{};
  // Initialize
  for (long i{0}; i < sequence_size; ++i) {
    data.push_back(i);
  }
  long result = 0L;
  auto ex = grppi::parallel_execution_sycl(std::stoi(argv[2]), std::stoi(argv[3]));
  auto t1 = std::chrono::system_clock::now();
  auto b_data = sycl::buffer(data.data(), sycl::range<1>(data.size()));
  auto b_result = sycl::buffer{&result, sycl::range<1>(1)};
  grppi::sycl_kernel::reduce(ex.get_queue(), data.size(), b_data, b_result, 0L, [](auto x, auto y) { return x+y; });
  auto t2 = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
  std::cout << sequence_size  << ", " << diff.count() << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Invalid number of arguments." << "\n";
    return -1;
  }
  if ("sycldir" == std::string(argv[1])) {
    direct(argv);
    return 0;
  }
  unsigned sequence_size(std::stol(argv[4]));
  std::vector<long> data{};
  // Initialize
  for (long i{0}; i < sequence_size; ++i) {
    data.push_back(i);
  }

  grppi::dynamic_execution ex = select_backend(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
  auto t1 = std::chrono::system_clock::now();
  grppi::reduce(ex, begin(data), end(data), 0L,[](auto x, auto y) { return x+y; });
  auto t2 = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
  std::cout << sequence_size  << ", " << diff.count() << std::endl;

  return 0;
}