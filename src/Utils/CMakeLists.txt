add_library(libutils INTERFACE)
target_include_directories(libutils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} fmt::fmt)
target_compile_features(libutils INTERFACE cxx_std_20)