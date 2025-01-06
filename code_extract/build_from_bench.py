import re
from pathlib import Path
import subprocess as sb
import os

def search_kernel_functions(directory_path):
    directory = Path(directory_path)
    file_exts = ['*.cu', '*.cpp', '*.h']
    
    func_pattern = re.compile(r'\b__global__\s+\w+\s+(\w+)\s*\(.*?\)\s*\{', re.DOTALL)
    function_names = {}

    for ext in file_exts:
        for file in directory.rglob(ext):
            with open(file, 'r', encoding='utf8', errors='ignore') as f:
                matches = func_pattern.findall(f.read())
                if matches:
                    function_names[Path(file)] = set(matches)

    return function_names

directory = '/home/grimmy/Mneme/code_extract/tests/HeCBench_cuda'  

kernel_functions = search_kernel_functions(directory)

os.chdir('/home/grimmy/Mneme/code_extract/build/temp')

for file, funcs in kernel_functions.items():
    print(funcs)
    cc_path = file.parent / 'compile_commands.json'
    if not cc_path.exists():
        cmd = ['bear', '--', 'make']
        os.chdir(file.parent)
        sb.run(cmd)
        os.chdir('/home/grimmy/Mneme/code_extract/build/temp')
    for func in funcs:
        cmd = ['../code-extract', file.parent, func]
        sb.run(cmd)
    break