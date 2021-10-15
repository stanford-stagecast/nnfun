add_executable (forward_propagation "forward_propagation.cc")
target_link_libraries (forward_propagation util)
#target_link_libraries (forward_propagation nn)
target_link_libraries ("forward_propagation" ${Eigen_LDFLAGS})
target_link_libraries ("forward_propagation" ${Eigen_LDFLAGS_OTHER})

add_executable (neuralnetwork "neuralnetwork.cc")
