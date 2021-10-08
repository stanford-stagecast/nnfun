add_executable (example1 "example1.cc")
target_link_libraries (example1 util nn)

target_link_libraries ("example1" ${Eigen_LDFLAGS})
target_link_libraries ("example1" ${Eigen_LDFLAGS_OTHER})

add_executable (neuralnetwork "neuralnetwork.cc")
target_link_libraries (neuralnetwork util nn)

target_link_libraries ("neuralnetwork" ${Eigen_LDFLAGS})
target_link_libraries ("neuralnetwork" ${Eigen_LDFLAGS_OTHER})
