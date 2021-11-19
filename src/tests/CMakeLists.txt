macro (add_test_exec exec_name)
    add_executable ("${exec_name}" "${exec_name}.cc")
    target_link_libraries ("${exec_name}" util ${ARGN})
endmacro (add_test_exec)

add_test_exec (eigentest1)
add_test_exec (formulagradienttest1 ${Eigen_LDFLAGS} ${Eigen_LDFLAGS_OTHER})
