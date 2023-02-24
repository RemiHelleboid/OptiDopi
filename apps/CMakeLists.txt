# add_executable(spad simple_spad.cpp)
# target_link_libraries(spad libdevice)
# target_link_libraries(spad fmt::fmt)

# add_executable(BVSweep spad_bv_sweep.cpp)
# target_link_libraries(BVSweep libdevice libutils)
# target_link_libraries(BVSweep fmt::fmt OpenMP::OpenMP_CXX)

# add_executable(test_optim test_optim.cpp)
# target_link_libraries(test_optim libdevice libutils liboptim)
# target_link_libraries(test_optim fmt::fmt OpenMP::OpenMP_CXX)

# add_executable(optim_spad_simple optim_spad_simple.cpp)
# target_link_libraries(optim_spad_simple libdevice libutils liboptim)
# target_link_libraries(optim_spad_simple fmt::fmt OpenMP::OpenMP_CXX)

# add_executable(test_interp test_interp.cpp)
# target_link_libraries(test_interp libdevice libutils liboptim)
# target_link_libraries(test_interp fmt::fmt OpenMP::OpenMP_CXX)

# add_executable(pswar_spad_simple pswar_spad_simple.cpp)
# target_link_libraries(pswar_spad_simple libdevice libutils liboptim)
# target_link_libraries(pswar_spad_simple fmt::fmt OpenMP::OpenMP_CXX)


# add_executable(spad_cost_function spad_cost_function.cpp)
# target_link_libraries(spad_cost_function libdevice libutils liboptim)
# target_link_libraries(spad_cost_function fmt::fmt OpenMP::OpenMP_CXX)

add_executable(PSWARM_SPAD.x PSWARM_SPAD.cpp)
target_link_libraries(PSWARM_SPAD.x fmt::fmt OpenMP::OpenMP_CXX)
target_link_libraries(PSWARM_SPAD.x libdevice libutils liboptim)

add_executable(SA_SPAD.x SA_SPAD.cpp)
target_link_libraries(SA_SPAD.x fmt::fmt OpenMP::OpenMP_CXX)
target_link_libraries(SA_SPAD.x libdevice libutils liboptim)
