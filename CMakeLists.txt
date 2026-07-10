cmake_minimum_required(VERSION 3.20)

project(MultiDICNative LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(native/ncorr)
add_subdirectory(native/recon3d)
