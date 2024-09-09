import sys
import os
import json
import subprocess as sp
import argparse
from pathlib import Path
import tempfile

def comma_separated_list(values):
    return values.split(',')


def replay(replay_bin, json_path:str, BlockDimx:int, GridDimx:int):
    with open(json_path, "r") as fd:
        recorded_data = json.load(fd)

    for kernel in recorded_data["Kernels"].keys():
        cmd = f"{replay_bin} --record-replay-json {json_path} --kernel-name {kernel} --blockDimx {BlockDimx} --gridDimx {GridDimx}"
        print("Going to run the following command:")
        print(f"{cmd}")
        try:
            result = sp.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )
        except sp.CalledProcessError as e:
            print(f"Error: {e.stderr}")
            print(f"stdout : {e.output}")
            print("Return code {e.returncode}")
            return e.returncode
        print(f"Replay kernel {kernel} stdout:")
        print(result.stdout)
    return 0


def execute_record(preload_lib:str, temp_dir: str, record_file:str, cmd:str):
    new_env = os.environ.copy()
    new_env["LD_PRELOAD"] = preload_lib
    new_env["RR_FILE"] = record_file 
    new_env["RR_DATA_DIR"] = temp_dir
    print("Going to run the following command:")
    print(f"LD_PRELOAD={preload_lib} RR_DATA_DIR={temp_dir} RR_FILE={record_file} {cmd}")
    try:
        result = sp.run(
            cmd,
            env = new_env,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )
    except sp.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        print(f"stdout : {e.output}")
        print(f"Return code {e.returncode}")
        return e.returncode
    print(result.stdout)
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="A script to test end to end record replay experiments"
    )

    parser.add_argument("--exe", "-e", help="The executable(test) to record and replay", required=True, type=str)
    parser.add_argument("--record-lib", "-r", help="Path to the librecord libray", required=True, type=str)
    parser.add_argument("--replay-bin", "-b", help="Path to the replay binary", required=True, type=str)
    parser.add_argument("--blocks", "-bd", help="Path to the replay binary", default=[-1], type=comma_separated_list)
    parser.add_argument("--grids", "-gd", help="Path to the replay binary", default=[-1], type=comma_separated_list)
    parser.add_argument("--args", "-a", help="Arguments to the test executable", type=comma_separated_list, default=[])

    args = parser.parse_args()

    test_executable=Path(args.exe)
    assert test_executable.exists(), f"Executable {test_executable} does not exist"

    preload_lib = Path(args.record_lib)
    assert preload_lib.exists(), f"Path to Preloading record lib {preload_lib} does not exist "

    replay_bin = Path(args.replay_bin)
    assert replay_bin.exists(), f"Path {replay_bin} to replay tool does not exist"

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        cmd = f"{test_executable} {' '.join(args.args)}"
        ret = execute_record(str(preload_lib), temp_dir_str, "record_replay_test.json", cmd) 
        if ret != 0:
            raise RuntimeError("Return code of record is not 0")

        record_json = temp_dir / Path("record_replay_test.json")

        assert record_json.exists(), f"Recorded file {record_json} does not exist"
        
        # All of our tests are thread invariant. Verify that we can modify thread ids.
        for blockDimx in args.blocks:
            for GridDimx in args.grids:
                if replay(replay_bin, str(record_json), blockDimx, GridDimx ) != 0:
                    raise RuntimeError("Replaying failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
