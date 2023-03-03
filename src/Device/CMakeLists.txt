set(HEADER_FILES_LIBDEVICE "DopingProfile1D.hpp" "Device1D.hpp" "PoissonSolver1D.hpp" "McIntyre.hpp" "AdvectionDiffusionMC.hpp" "ParticleAdvectionDiffusionMC.hpp")
set(SOURCE_FILES_LIBDEVICE "DopingProfile1D.cpp" "Device1D.cpp" "PoissonSolver1D.cpp" "McIntyre.cpp" "AdvectionDiffusionMC.cpp")


add_library(libdevice STATIC ${SOURCE_FILES_LIBDEVICE} ${HEADER_FILES_LIBDEVICE})
target_include_directories(libdevice PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(libdevice PUBLIC libutils libphysics)
target_link_libraries(libdevice PUBLIC Eigen3::Eigen)
target_link_libraries(libdevice PUBLIC fmt::fmt)
target_link_libraries(libdevice PUBLIC OpenMP::OpenMP_CXX)

target_compile_features(libdevice PRIVATE cxx_std_20)

