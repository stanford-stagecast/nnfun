macro (add_frontend_exec exec_name)
    add_executable ("${exec_name}" "${exec_name}.cc")
    target_link_libraries ("${exec_name}" util ${ARGN})
    target_link_libraries ("${exec_name}" ${Eigen_LDFLAGS})
    target_link_libraries ("${exec_name}" ${Eigen_LDFLAGS_OTHER})
endmacro (add_frontend_exec)

add_frontend_exec (forward_propagation)

add_executable (neuralnetwork "neuralnetwork.cc")
