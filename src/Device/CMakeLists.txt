set(HEADER_FILES_LIBDEVICE "doping_profile.hpp" "device.hpp" "PoissonSolver.hpp")

set(SOURCE_FILES_LIBDEVICE "doping_profile.cpp" "device.cpp" "PoissonSolver.cpp")


add_library(libdevice STATIC ${SOURCE_FILES_LIBDEVICE} ${HEADER_FILES_LIBDEVICE})
target_include_directories(libdevice PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libdevice PUBLIC libutils)
target_link_libraries(libdevice PUBLIC Eigen3::Eigen)
target_link_libraries(libdevice PUBLIC fmt::fmt)

target_compile_features(libdevice PRIVATE cxx_std_20)

