cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME_STR grppi)
project(${PROJECT_NAME_STR})

# Path for additional CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules)

# Print processor and compiler 
message(STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR})
message(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})

if (NOT DEFINED CMAKE_CXX_STANDARD)
  message(STATUS "C++ standard not set")
  set(CMAKE_CXX_STANDARD 17)
endif ()
if (CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "Minimum standard is C++17")
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard set to ${CMAKE_CXX_STANDARD}")

add_compile_options(-Wall -Werror -pedantic -pedantic-errors -Wextra)

# Set specific options depending on compiler
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  if (NOT (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 3.9.0))
    message(FATAL_ERROR "Clang version " ${CMAKE_CXX_COMPILER_VERSION}
        " not supported. Upgrade to 3.9 or above.")
  endif ()
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  if (NOT (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 6.0))
    message(FATAL_ERROR "g++ version " ${CMAKE_CXX_COMPILER_VERSION}
        " not supported. Upgrade to 6.0 or above.")
  else ()
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 7.0)
      #g++ 7 warns in non C++17 for over-aligned new otherwise
      add_compile_options(-faligned-new)
    endif ()
  endif ()
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  message(WARNING "Intel compiler is not currently supported")
  # if(NOT(${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 16.0))
  #   message( FATAL_ERROR "Intel version " ${CMAKE_CXX_COMPILER_VERSION}
  #       " not supported. Upgrade to 16.0 or above.")
  # else()
  #   message( STATUS "C++ Compiler is Intel" )
  #   message( WARNING "WARNING: GrPPI has not been fully tested with Intel compiler." )
  # endif()
endif ()

message(STATUS "C++ Compiler is " ${CMAKE_CXX_COMPILER_ID}
    " version " ${CMAKE_CXX_COMPILER_VERSION})

# GrPPI library
include_directories(BEFORE SYSTEM "${CMAKE_SOURCE_DIR}/include")

# TBB Library
find_package(TBB)

if (TBB_FOUND)
  if (${CMAKE_CXX_STANDARD} STREQUAL 20)
    option(GRPPI_TBB_ENABLE "Require TBB library" OFF)
    set(GRPPI_TBB_ENABLE OFF CACHE BOOL "Require TBB library" FORCE)
    message(STATUS "TBB found, but not supported in C++20")
  else ()
    option(GRPPI_TBB_ENABLE "Require TBB library" ON)
    message(STATUS "TBB found")
  endif ()
else (TBB_FOUND)
  option(GRPPI_TBB_ENABLE "Require TBB library" OFF)
  set(GRPPI_TBB_ENABLE OFF CACHE BOOL "Require TBB library" FORCE)
  message(STATUS "TBB not found")
endif (TBB_FOUND)

message(STATUS "TBB: " ${GRPPI_TBB_ENABLE})
if (GRPPI_TBB_ENABLE)
  link_libraries(TBB::tbb)
  add_definitions(-DGRPPI_TBB)
endif (GRPPI_TBB_ENABLE)

# Threads Library
find_package(Threads REQUIRED)

# OMP library
find_package(OpenMP)
if (OPENMP_FOUND)
  option(GRPPI_OMP_ENABLE "Require OpenMP library" ON)
  message(STATUS "OpenMP found")
else (OPENMP_FOUND)
  option(GRPPI_OMP_ENABLE "Require OpenMP library" OFF)
  set(GRPPI_OMP_ENABLE OFF CACHE BOOL "Require OpenMP library" FORCE)
  message(STATUS "OpenMP not found")
endif (OPENMP_FOUND)

message(STATUS "OpenMP: " ${GRPPI_OMP_ENABLE})
if (GRPPI_OMP_ENABLE)
  add_definitions(-DGRPPI_OMP)
  add_compile_options(${OpenMP_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif (GRPPI_OMP_ENABLE)

#FastFlow library
find_package(FF)
if (FF_FOUND)
  if (${CMAKE_CXX_STANDARD} STREQUAL "17")
    option(GRPPI_FF_ENABLE "Require FastFlow library" OFF)
    set(GRPPI_FF_ENABLE OFF CACHE BOOL "Require FastFlow library" FORCE)
    message(STATUS "FastFlow not supported in C++17")
  else ()
    option(GRPPI_FF_ENABLE "Require FastFlow library" ON)
    message(STATUS "FastFlow found")
  endif ()
else ()
  option(GRPPI_FF_ENABLE "Require FastFlow library" OFF)
  set(GRPPI_FF_ENABLE OFF CACHE BOOL "Require FastFlow library" FORCE)
  message(STATUS "FastFlow not found")
  message(STATUS "  GRPPI_FF_ENABLE has been set to OFF")
  message(STATUS "  To enable it, please, first install FastFlow")
  message(STATUS "  https://github.com/fastflow/fastflow")
endif ()

message(STATUS "FastFlow: " ${GRPPI_FF_ENABLE})
if (GRPPI_FF_ENABLE)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(GRPPI_FF_ENABLE OFF CACHE BOOL "Require Fastflow library" FORCE)
    message(WARNING "Fastflow is not currently supported in Clang")
  else ()
    add_definitions(-DGRPPI_FF)
    include_directories(${FF_INCLUDE_DIRS})
  endif ()
endif (GRPPI_FF_ENABLE)

# SYCL Library
set(SYCL_LANGUAGE_VERSION 202001 CACHE STRING "" FORCE)
# Intel Compiler
if (${CMAKE_CXX_COMPILER} MATCHES "/opt/intel/oneapi/compiler/(.)+/linux/bin/dpcpp")
  set(GRPPI_SYCL_DPCPP_FOUND ON CACHE INTERNAL "DPC++ compiler found")
  message(STATUS "SYCL: DPC++ found")
else()
  set(GRPPI_SYCL_DPCPP_FOUND OFF CACHE INTERNAL "DPC++ compiler found")
endif()
# ComputeCpp Compiler
find_package(ComputeCpp)
if (COMPUTECPP_FOUND)
  message(STATUS "SYCL: ComputeCpp found")
endif()
# SYCL Compiler
if (GRPPI_SYCL_DPCPP_FOUND OR COMPUTECPP_FOUND)
  option(GRPPI_SYCL_ENABLE "Require SYCL compiler" ON)
else()
  option(GRPPI_SYCL_ENABLE "Require SYCL compiler" OFF)
  set(GRPPI_SYCL_ENABLE OFF CACHE BOOL "Require SYCL compiler" FORCE)
  message(STATUS "SYCL compiler not found")
endif()
# Multiple compiler warning
if (GRPPI_SYCL_DPCPP_FOUND AND COMPUTECPP_FOUND)
  message(WARNING "SYCL: Multiple compilers found. This may cause redefinition errors")
endif()

message(STATUS "SYCL Compiler: " ${GRPPI_SYCL_ENABLE})
if(GRPPI_SYCL_ENABLE)
  add_definitions(-DGRPPI_SYCL)
  if (COMPUTECPP_FOUND)
    include_directories(${ComputeCpp_INCLUDE_DIRS})
  endif ()
  #TODO Improve
  set(CL_TARGET_OPENCL_VERSION 220)
  add_compile_definitions(CL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION})
  #add_compile_options(-fsycl-targets=amdgcn-unknown-unknown-sycldevice)
  #add_compile_options(-fPIC)
endif()


# Doxygen Documentation
option(GRPPI_DOXY_ENABLE "Require Doxygen library" OFF)
if (GRPPI_DOXY_ENABLE)
  add_subdirectory(doxy)
endif (GRPPI_DOXY_ENABLE)

# Installation
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/${CMAKE_PROJECT_NAME}"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.h")

# Example applications
option(GRPPI_EXAMPLE_APPLICATIONS_ENABLE "Sample tests" OFF)
if (GRPPI_EXAMPLE_APPLICATIONS_ENABLE)
  add_subdirectory(samples)
endif (GRPPI_EXAMPLE_APPLICATIONS_ENABLE)

# Unit Tests
enable_testing()
option(GRPPI_UNIT_TEST_ENABLE "Unit tests" ON)
if (GRPPI_UNIT_TEST_ENABLE)
  add_subdirectory(unit_tests)
endif (GRPPI_UNIT_TEST_ENABLE)
