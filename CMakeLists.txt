cmake_minimum_required ( VERSION 2.8.5 )
cmake_policy ( SET CMP0048 NEW )
project ( nnfun )

include ( etc/build_type.cmake )
include ( etc/cflags.cmake )
include ( etc/clang_format.cmake )
include ( etc/tests.cmake )

find_package ( PkgConfig )

pkg_check_modules ( Eigen REQUIRED eigen3 )
include_directories ( SYSTEM ${Eigen_INCLUDE_DIRS} )
add_compile_options ( ${Eigen_CFLAGS} )

include_directories ("${PROJECT_SOURCE_DIR}/src/util")
include_directories ("${PROJECT_SOURCE_DIR}/src/audio")
include_directories ("${PROJECT_SOURCE_DIR}/src/nn")

add_subdirectory ("${PROJECT_SOURCE_DIR}/src/util")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/audio")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/frontend")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/tests")
add_subdirectory ("${PROJECT_SOURCE_DIR}/src/nn")

add_subdirectory ("${PROJECT_SOURCE_DIR}/src/frontend/linear_predictor")
