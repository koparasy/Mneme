import subprocess
import os
from tqdm import tqdm
from pathlib import Path

def print_red(*skk): print("\033[91m{}\033[00m" .format(" ".join(map(str, list(skk)))))
 
def print_green(*skk): print("\033[92m{}\033[00m" .format(" ".join(map(str, list(skk)))))

class TestTool():
    
    def __init__(self):
        self.total_tests = 0
        self.total_fails = 0
        self.project_dir = Path(os.path.abspath(__file__)).parent
        self.exec = self.project_dir / Path("build/code-extract")
        os.chdir(self.exec.parent)
        subprocess.run(['mkdir', '-p', 'test'], check=True)
        os.chdir('test')

    def run_tool(self, *args):
        result = subprocess.run(
            [self.exec] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True                    
        )
        return result

    def run_single_test_set(self, project, funcs, test_name):
        fails = 0
        for func in tqdm(funcs):
            result = self.run_tool(project, func)
            msg = "\nTraceback for " + func + "\n" + result.stderr
            if 'Compilation successful!' not in result.stdout:
                print_red(test_name, "failed with", func, "!")
                print(msg)
                fails += 1
        if fails:
            self.total_fails += 1
            print_red(test_name, " failed! (", len(funcs) - fails, "/", len(funcs), "Passed)")
        else: 
            print_green(test_name, " passed! (", len(funcs), "/", len(funcs), "Passed)")
        self.total_tests += 1
        return fails

    def print_start(self):
        print("\n=============================")
        print("   Starting Tests")
        print("=============================\n")

    def print_end(self):
        print("\n=============================")
        print("   Test Results Summary")
        print("=============================")
        print(f"Total tests run: {self.total_tests}")
        print_green(f"  Passed: {self.total_tests - self.total_fails}")
        print_red(f"  Failed: {self.total_fails}")
        print("=============================")

    def run_tests(self):
        # Simple - test simple function bodies 
        path_to_proj = self.project_dir / Path("tests/dummy_proj/build")
        function_names = ['func'] + ['func' + str(i) for i in range(2, 10 + 1)]
        self.run_single_test_set(path_to_proj, function_names, "Simple")
        
    def run(self):
        self.print_start()
        self.run_tests()
        self.print_end()

if __name__ == '__main__':
    tester = TestTool()
    tester.run()