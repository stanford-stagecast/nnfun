macro (add_benchmark_exec exec_name)
    add_executable ("${exec_name}" "${exec_name}.cc")
    target_link_libraries ("${exec_name}" util ${ARGN})
    target_link_libraries ("${exec_name}" ${Eigen_LDFLAGS})
    target_link_libraries ("${exec_name}" ${Eigen_LDFLAGS_OTHER})
endmacro (add_benchmark_exec)

add_benchmark_exec (benchmark_1)
add_benchmark_exec (benchmark_2)
add_benchmark_exec (benchmark_3)
add_benchmark_exec (benchmark_4)
add_benchmark_exec (benchmark_5)
add_benchmark_exec (benchmark_6)

