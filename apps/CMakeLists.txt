add_executable(spad simple_spad.cpp)
target_link_libraries(spad libdevice)
target_link_libraries(spad fmt::fmt)

add_executable(BVSweep spad_bv_sweep.cpp)
target_link_libraries(BVSweep libdevice libutils)
target_link_libraries(BVSweep fmt::fmt OpenMP::OpenMP_CXX)