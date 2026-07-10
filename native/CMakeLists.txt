cmake_minimum_required(VERSION 3.20)

project(MultiDICNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ncorr)
add_subdirectory(recon3d)
