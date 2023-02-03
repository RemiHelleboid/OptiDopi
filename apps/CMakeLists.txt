add_executable(spad simple_spad.cpp)
target_link_libraries(spad libdevice)
target_link_libraries(spad fmt::fmt)