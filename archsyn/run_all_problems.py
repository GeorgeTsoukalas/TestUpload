import subprocess

file_name = "new_main.py"

for x in range(1, 134): # the problems in the code2inv benchmark
    arg = "--batch_size 50 --random_seed 0 -data_root ./benchmarks -file_list all -single_sample numerical_invariant_depth2.sl -top_left True -GM True --sem minmax --problem_num " + str(problem_num)
    subprocess.Popen(["python", file_name, arg])