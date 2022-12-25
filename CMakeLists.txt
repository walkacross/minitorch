cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(minitorch)



# ================================================================
# prerequist
# ================================================================

# cxx compile standard
set(CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# option for unittest switch
option(ENABLE_UNITTEST "unittest for libminimal" OFF)

# project version setup
set(MINITORCH_MAJOR_VERSION "0")
set(MINITORCH_MINOR_VERSION "0")
set(MINITORCH_PATCH_VERSION "1")

# debug message
message(STATUS "======================================================")
message(STATUS "minitorch release version: ${MINITORCH_MAJOR_VERSION}.${MINITORCH_MINOR_VERSION}.${MINITORCH_PATCH_VERSION}")
message(STATUS "minitorch unittest flag: ${ENABLE_UNITTEST}")
message(STATUS "======================================================")

# generate library
add_subdirectory(c10)
