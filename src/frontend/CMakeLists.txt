add_executable (example1 "example1.cc")
target_link_libraries (example1 util nn)

target_link_libraries ("example1" ${Eigen_LDFLAGS})
target_link_libraries ("example1" ${Eigen_LDFLAGS_OTHER})

add_executable (forward_propagation "forward_propagation.cc")
add_executable (neuralnetwork "neuralnetwork.cc")
