macro (add_test_exec exec_name)
    add_executable ("${exec_name}" "${exec_name}.cc")
    target_link_libraries ("${exec_name}" util ${Eigen_LDFLAGS} ${Eigen_LDFLAGS_OTHER})
endmacro (add_test_exec)

add_test_exec (simple-test)
add_test_exec (test_boundary)
add_test_exec (complex-test)
add_test_exec (compare_eta)
