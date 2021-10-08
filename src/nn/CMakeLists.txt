add_executable (neuralnetwork "neuralnetwork.cc")
target_link_libraries (neuralnetwork util)

target_link_libraries ("neuralnetwork" ${Eigen_LDFLAGS})
target_link_libraries ("neuralnetwork" ${Eigen_LDFLAGS_OTHER})
