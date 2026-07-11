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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  target_compile_options(native_ncorr PRIVATE -march=x86-64 -mtune=generic -Wa,-mx86-used-note=no)
endif()

add_executable(ncorr_cli
  tools/ncorr_cli.cpp
)
target_link_libraries(ncorr_cli PRIVATE native_ncorr)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  target_compile_options(ncorr_cli PRIVATE -march=x86-64 -mtune=generic -Wa,-mx86-used-note=no)
endif()

install(TARGETS ncorr_cli
  RUNTIME DESTINATION pymultidic/bin
)
