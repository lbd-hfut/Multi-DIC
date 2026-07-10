add_library(native_ncorr STATIC
  src/ncorr_api.cpp
)

target_include_directories(native_ncorr
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
  target_compile_options(native_ncorr PRIVATE /W4)
else()
  target_compile_options(native_ncorr PRIVATE -Wall -Wextra -Wpedantic)
endif()
