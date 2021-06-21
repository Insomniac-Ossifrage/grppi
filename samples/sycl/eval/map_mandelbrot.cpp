#ifndef GRPPI_SAMPLES_SYCL_MANDELBROT_H
#define GRPPI_SAMPLES_SYCL_MANDELBROT_H

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

  void direct(char **argv) {
    auto ex = grppi::parallel_execution_sycl(std::stoi(argv[2]), std::stoi(argv[3]));
    unsigned image_size(std::stoul(argv[4]));
    unsigned iterations(std::stoul(argv[5]));
    double xmin = -1.0;
    double xmax = 1.0;
    double ymin = -1.0;
    double ymax = 1.0;
    std::vector<double> data{};
    std::vector<double> pixel_pos{};
    // Initialize
    for (unsigned long i{0}; i < image_size * image_size; ++i) {
      data.push_back(0);
    }
    for (unsigned long i{0}; i < image_size; i++) {
      pixel_pos.push_back(ymin + i * (ymax - ymin) / image_size);
    }
    // Multiple Launches
    auto t1 = std::chrono::system_clock::now();
    auto b_pixel = sycl::buffer(pixel_pos.data(), sycl::range<1>(pixel_pos.size()));
    b_pixel.set_write_back(false);
    b_pixel.set_final_data(nullptr);
    for (unsigned i{0}; i < image_size; i++) {
      auto b_data = sycl::buffer(data.data() + i * image_size, sycl::range<1>(pixel_pos.size()));
      double point = xmin + i * (xmax - xmin) / image_size;
      std::array buffer = {b_pixel};
      grppi::sycl_kernel::map(ex.get_queue(), image_size, buffer, b_data, [=](auto elem) {
          std::complex<double> z{0};
          std::complex<double> alpha(point, elem);

          unsigned j = 0;
          while (std::abs(z) < 10 && j < iterations) {
            z = std::pow(z, 2) + alpha;
            j++;
          }
          return j;
      });
    }
    auto t2 = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << image_size << ", " << iterations << ", " << diff.count() << std::endl;

    std::ofstream outfile("mandelbrot.pbm");
    outfile << "P2 " << image_size << " " << image_size << " " << iterations << "\n";
    // Save file
    for (unsigned i{0}; i < image_size; i++) {
      for (unsigned j{0}; j < image_size; j++) {
        outfile << (int) data[i * image_size + j] << " ";
      }
      outfile << "\n";
    }
    outfile.close();
  }

  int main(int argc, char **argv) {
    if (argc < 6) {
      std::cout << "Invalid number of arguments." << "\n";
      return -1;
    }
    if ("sycldir" == std::string(argv[1])) {
      direct(argv);
      return 0;
    }
    grppi::dynamic_execution ex = select_backend(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));
    unsigned image_size(std::stoul(argv[4]));
    unsigned iterations(std::stoul(argv[5]));
    double xmin = -1.0;
    double xmax = 1.0;
    double ymin = -1.0;
    double ymax = 1.0;
    std::vector<double> data{};
    std::vector<double> pixel_pos{};
    // Initialize
    for (unsigned long i{0}; i < image_size * image_size; ++i) {
      data.push_back(0);
    }
    for (unsigned long i{0}; i < image_size; i++) {
      pixel_pos.push_back(ymin + i * (ymax - ymin) / image_size);
    }
    // Multiple Launches
    auto t1 = std::chrono::system_clock::now();
    for (unsigned i{0}; i < image_size; i++) {
      auto data_iter_b = data.begin() + i * image_size;
      double point = xmin + i * (xmax - xmin) / image_size;
      grppi::map(ex, pixel_pos.begin(), pixel_pos.end(), data_iter_b, [=](auto elem) {
          std::complex<double> z{0};
          std::complex<double> alpha(point, elem);

          unsigned j = 0;
          while (std::abs(z) < 10 && j < iterations) {
            z = std::pow(z, 2) + alpha;
            j++;
          }
          return j;
      });
    }
    auto t2 = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << image_size << ", " << iterations << ", " << diff.count() << std::endl;

    std::ofstream outfile("mandelbrot.pbm");
    outfile << "P2 " << image_size << " " << image_size << " " << iterations << "\n";
    // Save file
    for (unsigned i{0}; i < image_size; i++) {
      for (unsigned j{0}; j < image_size; j++) {
        outfile << (int) data[i * image_size + j] << " ";
      }
      outfile << "\n";
    }
    outfile.close();
    return 0;
  }

#endif