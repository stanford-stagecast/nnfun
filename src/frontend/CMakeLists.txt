add_executable (example1 "example1.cc")
target_link_libraries (example1 util)

target_link_libraries ("example1" ${Eigen_LDFLAGS})
target_link_libraries ("example1" ${Eigen_LDFLAGS_OTHER})
