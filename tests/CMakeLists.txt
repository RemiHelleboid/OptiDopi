file(GLOB BENCHMARKS *.cpp)

message("OptiDopi TESTS: ")

foreach(BENCH ${BENCHMARKS})
    get_filename_component(EXAMPLENAME ${BENCH} NAME_WE)
    message("   --- Test ${EXAMPLENAME} added.")
    add_executable(${EXAMPLENAME} ${BENCH})
    target_link_libraries(${EXAMPLENAME} PRIVATE libdevice libutils liboptim  doctest::doctest)
    add_test(${EXAMPLENAME} ${EXAMPLENAME})
endforeach()