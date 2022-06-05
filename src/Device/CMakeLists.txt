set(HEADER_FILES_LIBDEVICE "doping_profile.hpp" "device.hpp")

set(SOURCE_FILES_LIBDEVICE "doping_profile.cpp" "device.cpp")

add_library(libdevice STATIC ${SOURCE_FILES_LIBDEVICE} ${HEADER_FILES_LIBDEVICE})

target_link_libraries(libdevice PUBLIC libutils)

target_compile_features(libdevice PRIVATE cxx_std_20)

