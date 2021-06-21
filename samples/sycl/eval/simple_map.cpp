#include <iostream>
#include <complex>
#include <vector>
#include "grppi/grppi.h"
#include "grppi/dyn/dynamic_execution.h"

grppi::dynamic_execution select_backend(unsigned platform, unsigned device) {
  return grppi::parallel_execution_sycl{platform, device};
}

void direct(std::vector<long> &vector1, std::vector<long> &vector2, std::vector<long> &vector3, unsigned long sequence_size) {
  grppi::parallel_execution_sycl ex{2,0};

  auto t1 = std::chrono::system_clock::now();
  sycl::buffer<long, 1> b1 = sycl::buffer(vector1.data(), sycl::range<1>(sequence_size));
  b1.set_write_back(false);
  b1.set_final_data(nullptr);
  sycl::buffer<long, 1> b2 = sycl::buffer(vector2.data(), sycl::range<1>(sequence_size));
  b2.set_write_back(false);
  b2.set_final_data(nullptr);
  sycl::buffer<long, 1> b3 = sycl::buffer(vector3.data(), sycl::range<1>(sequence_size));

  std::array<sycl::buffer<long,1>,2> buffers = {b1, b2};
  grppi::sycl_kernel::map(ex.get_queue(), sequence_size, buffers, b3, [=](auto a, auto b) {
      return a+b;
  });

  auto t2 = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << sequence_size << ", " << diff.count() << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Invalid number of arguments." << "\n";
  }
  unsigned long sequence_size = std::stoul(argv[2]);
  std::vector<long> vector1{};
  std::vector<long> vector2{};
  std::vector<long> vector3{};
  vector1.reserve(sequence_size);
  vector2.reserve(sequence_size);
  vector3.reserve(sequence_size);

  generate_n(back_inserter(vector1), static_cast<long>(sequence_size),
             [i=0]() mutable { return i++; });
  generate_n(back_inserter(vector2), static_cast<long>(sequence_size),
             [i=0]() mutable { return i++; });
  generate_n(back_inserter(vector3), static_cast<long>(sequence_size),
             []() { return 0; });

  if ("sycldir" == std::string(argv[1])) {
    direct(vector1, vector2, vector3, sequence_size);
    return 0;
  }

  auto ex = select_backend(2,0);
  auto t1 = std::chrono::system_clock::now();
  grppi::map(ex, std::make_tuple(vector1.begin(), vector2.begin()), vector1.end() ,vector3.begin(), [=] (auto a, auto b) {
    return a+b;
  });
  auto t2 = std::chrono::system_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << sequence_size << ", " << diff.count() << std::endl;

  return 0;
}

