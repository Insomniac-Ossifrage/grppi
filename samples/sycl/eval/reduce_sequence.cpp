#include <iostream>
#include <complex>
#include <vector>
#include "grppi/grppi.h"
#include "grppi/dyn/dynamic_execution.h"


grppi::dynamic_execution select_backend(const std::string &mode, unsigned platform, unsigned device) {
  if ("seq" == mode) return grppi::sequential_execution{};
  if ("sycl" == mode) return grppi::parallel_execution_sycl{platform, device};
  return {};
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cout << "Invalid number of arguments." << "\n";
    return -1;
  }

  grppi::dynamic_execution ex = select_backend(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));

  return 0;
}