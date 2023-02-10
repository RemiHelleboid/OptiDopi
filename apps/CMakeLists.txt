add_executable(spad simple_spad.cpp)
target_link_libraries(spad libdevice)
target_link_libraries(spad fmt::fmt)

add_executable(BVSweep spad_bv_sweep.cpp)
target_link_libraries(BVSweep libdevice libutils)
target_link_libraries(BVSweep fmt::fmt OpenMP::OpenMP_CXX)


add_executable(test_optim test_optim.cpp)
target_link_libraries(test_optim libdevice libutils liboptim)
target_link_libraries(test_optim fmt::fmt OpenMP::OpenMP_CXX)

add_executable(optim_spad_simple optim_spad_simple.cpp)
target_link_libraries(optim_spad_simple libdevice libutils liboptim)
target_link_libraries(optim_spad_simple fmt::fmt OpenMP::OpenMP_CXX)