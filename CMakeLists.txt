cmake_minimum_required(VERSION 3.18)
project(ckks_fl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build with aggressive optimizations by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(
  -O3
  -march=native
  -ffast-math
  # Activates #pragma omp simd (single-thread SIMD hints) without pulling in
  # the OpenMP threading runtime.  Full -fopenmp is reserved for when the NTT
  # butterfly loops are ready to be parallelised across cores.
  -fopenmp-simd
)

# ---------- Dependencies ----------

# pybind11 (only required when building Python bindings)
option(CKKS_BUILD_PYTHON "Build pybind11 Python extension" OFF)
if(CKKS_BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
endif()

# Eigen (optional; falls back to raw buffers if not found)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
  message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
  add_compile_definitions(CKKS_USE_EIGEN)
else()
  message(STATUS "Eigen3 not found — using custom buffers")
endif()

# ---------- Core C++ library ----------

add_library(ckks_core STATIC
  # Module 1 — Math utilities
  src/math_utils.cpp
  src/poly.cpp
  src/ntt.cpp

  # Module 2 — Sampling
  src/sampler.cpp
  # src/sampling/gaussian.cpp
  # src/sampling/uniform.cpp

  # Module 3 — CKKS core
  src/ckks_ops.cpp
  src/ckks/encoder.cpp
  src/keygen.cpp
  src/ckks_core.cpp

  # Module 4 — FL operations
  src/evaluator.cpp
  # src/fl/scalar_mul.cpp

  # Module 5 — Serialization (network layer)
  src/serialization.cpp
)

target_include_directories(ckks_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(ckks_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(Eigen3_FOUND)
  target_link_libraries(ckks_core PUBLIC Eigen3::Eigen)
endif()

# ---------- Python extension (Module 5) ----------

if(CKKS_BUILD_PYTHON)
  pybind11_add_module(ckks_python bindings/module.cpp)
  target_link_libraries(ckks_python PRIVATE ckks_core)
  install(TARGETS ckks_python DESTINATION .)
endif()

# ---------- Tests ----------

enable_testing()
add_subdirectory(tests/cpp)
