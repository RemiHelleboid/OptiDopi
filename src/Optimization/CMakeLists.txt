set(HEADER_FILES_LIBDEVICE  "SimulatedAnneal.hpp" "ParticleSwarm.hpp"  "MainFunctions.hpp")
set(SOURCE_FILES_LIBDEVICE "SimulatedAnneal.cpp" "ParticleSwarm.cpp" "MainFunctions.cpp")


add_library(liboptim STATIC ${SOURCE_FILES_LIBDEVICE} ${HEADER_FILES_LIBDEVICE})
target_include_directories(liboptim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(liboptim PUBLIC libutils libdevice)
target_link_libraries(liboptim PUBLIC Eigen3::Eigen)
target_link_libraries(liboptim PUBLIC fmt::fmt)
target_link_libraries(liboptim PUBLIC OpenMP::OpenMP_CXX)

target_compile_features(liboptim PRIVATE cxx_std_20)

