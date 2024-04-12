cmake_minimum_required(VERSION 3.15...3.23)

include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)

project(
  OptiDopti
  VERSION 0.0.0
  LANGUAGES CXX)

macro(opt OPTION HELP VALUE)
  option(ENABLE_${OPTION} ${HELP} ${VALUE})
  set(OPT_TEXI
      "${OPT_TEXI}\n@item ENABLE_${OPTION}\n${HELP} (default: ${VALUE})")
  message("Compilation option : " ${OPT_TEXI} "is set to : " ${VALUE})
endmacro()

function(enable_cxx_compiler_flag_if_supported flag)
  string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
  if(flag_already_set EQUAL -1)
    check_cxx_compiler_flag("${flag}" flag_supported)
    if(flag_supported)
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} ${flag}"
          PARENT_SCOPE)
    endif()
    unset(flag_supported CACHE)
  endif()
endfunction()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  set(default_build_type "Release")
  if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    set(default_build_type "RelWithDebInfo")
  endif()

  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(
      STATUS
        "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "${default_build_type}"
        CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                                 "MinSizeRel" "RelWithDebInfo")
  endif()

  # set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*)

  # Compiler tags
  enable_cxx_compiler_flag_if_supported("-Wall")
  enable_cxx_compiler_flag_if_supported("-Wextra")
  enable_cxx_compiler_flag_if_supported("-pedantic")
  enable_cxx_compiler_flag_if_supported("-Wshadow")
  enable_cxx_compiler_flag_if_supported("-Wnon-virtual-dtor")
  enable_cxx_compiler_flag_if_supported("-Wnon-virtual-dtor")
  enable_cxx_compiler_flag_if_supported("-Wno-deprecated-copy")
  enable_cxx_compiler_flag_if_supported("-fsanitize=address")

  

endif()

message("Optimization for the local machine is enabled")
# add_compile_options(-march=native)


find_package(Eigen3 NO_MODULE)

if(NOT Eigen3_FOUND)
  set(EIGEN_BUILD_DOC OFF)
  set(BUILD_TESTING OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  message("Fetching Eigen3 lib ...")
  add_compile_definitions(EIGEN_USE_LAPACKE=0)
  FetchContent_MakeAvailable(Eigen)
endif()

FetchContent_Declare(
  DocTest
  GIT_REPOSITORY "https://github.com/onqtam/doctest"
  GIT_TAG "4d8716f1efc1d14aa736ef52ee727bd4204f4c40"
)
FetchContent_MakeAvailable(DocTest)

# Find OpenMP
find_package(OpenMP REQUIRED)


# Fetch fmt library
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0)

find_package(fmt CONFIG QUIET)
if(NOT fmt_FOUND)
  message("Fetching fmt lib ...")
  FetchContent_MakeAvailable(fmt)
endif()




add_subdirectory(src/Physics)
add_subdirectory(src/Utils)
add_subdirectory(src/Device)
add_subdirectory(src/Optimization)
add_subdirectory(apps/)
add_subdirectory(tests/)