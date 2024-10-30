import argparse
import csv
import json
from pathlib import Path
import subprocess as sp
import re
import sys


def comma_separated_list(values):
    return [int(v) for v in values.split(",")]


def parse_replay_output(stdout: str):
    pattern = (
        r"Mean execution time \(ms\): ([\d\.]+) standard Deviation: ([\d\.]+) Block"
    )
    match = re.search(pattern, stdout)

    if not match:
        raise RuntimeError("Could not find execution time")

    ms = match.group(1)
    std = match.group(2)
    return float(ms), float(std)


def replay(
    replay_bin,
    json_path: str,
    kernel: str,
    BlockDimx: int,
    GridDimx: int,
    MaxThreads: int,
    MinBlocks: int,
):
    cmd = f"{replay_bin} --record-replay-json {json_path} --kernel-name {kernel} --blockDimx {BlockDimx} --gridDimx {GridDimx} --max-threads {MaxThreads} --min-blocks {MinBlocks} --repeat 5"
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
        raise e
    else:
        ms, std = parse_replay_output(result.stdout)
        return (ms, std, result.returncode)


def main():
    makefiles = {}
    replay_tool = "/usr/WS2/koparasy/RecordReplay/build_tioga/src/replay"
    parser = argparse.ArgumentParser(
        description="Grid search approach to identify optimal configuration settings"
    )

    parser.add_argument(
        "-j",
        "--json",
        type=str,
        help="path to the json database",
        required=True,
    )

    parser.add_argument(
        "-r",
        "--replay-path",
        type=str,
        help="path to the replay tool",
        required=True,
    )

    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        help="kernel name to optimize",
        required=True,
    )

    parser.add_argument(
        "--blocks",
        "-bd",
        help="Block dimensions to search for optimal configuration",
        default=[32, 64, 128, 256, 512, 1024],
        type=comma_separated_list,
    )
    parser.add_argument(
        "--grids",
        "-gd",
        help="Grid dimension to search for optimal configuration",
        default=[80, 160, 240, 320, 640],
        type=comma_separated_list,
    )
    parser.add_argument(
        "--csv", "-c", default="rr_opt.csv", help="File name of optimal configurations"
    )

    args = parser.parse_args()
    json_file = Path(args.json)
    assert json_file.exists(), f"JSON record db file ({json_file}) does not exist"

    replay_tool = args.replay_path
    assert Path(
        replay_tool
    ).exists(), "The provide path to the replay tool does not exist"
    DefaultBlockX = 256
    DefaultGridDimX = 1

    with open(str(json_file), "r") as fd:
        kernels = json.load(fd)
        assert (
            args.kernel in kernels["Kernels"]
        ), f"Kernel {args.kernel} does not exist in record database"
        DefaultBlockX = kernels["Kernels"][args.kernel]["Block"]["x"]
        DefaultGridDimX = kernels["Kernels"][args.kernel]["Grid"]["x"]

    results = []
    ms, std, verify = replay(
        replay_tool, args.json, args.kernel, DefaultBlockX, DefaultGridDimX, -1, -1
    )
    results.append(
        (args.kernel, DefaultGridDimX, DefaultBlockX, -1, -1, ms, std, verify)
    )
    baseline = ms

    print("GridDimX, BlockDimX, MaxThreads, MinBlocks, ms, std, verify")
    for GridDimX in args.grids:
        for BlockDimX in args.blocks:
            MinBlocksMax = int(1024 / BlockDimX)
            for MinBlocks in range(0, MinBlocksMax):
                ms, std, verify = replay(
                    replay_tool,
                    args.json,
                    args.kernel,
                    BlockDimX,
                    GridDimX,
                    BlockDimX,
                    MinBlocks + 1,
                )
                results.append(
                    (
                        args.kernel,
                        GridDimX,
                        BlockDimX,
                        BlockDimX,
                        MinBlocks + 1,
                        ms,
                        std,
                        verify,
                    )
                )
                print(GridDimX, BlockDimX, BlockDimX, MinBlocks + 1, ms, std, verify)

    best_config = min(results, key=lambda x: x[5])
    speedup = baseline / best_config[5]
    print(
        f"Optimal Configuration is GridDimX: {best_config[1]} BlockDimX: {best_config[2]} MaxThreads: {best_config[3]} MinBlockX: {best_config[4]} Execution Time: {best_config[5]} estimate speedup is: {speedup}"
    )

    # Define the header
    header = [
        "Kernel-Name",
        "GridDim-X",
        "BlockDim-X",
        "MaxThreads",
        "MinBlock",
        "Duration (ms)",
        "Duration (stdev)",
        "Verify (0 means correct)",
    ]

    # Writing to a CSV file
    csv_exists = Path(args.csv).exists()
    with open(args.csv, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        if not csv_exists:
            writer.writerow(header)

        # Write the data
        writer.writerows(results)


if __name__ == "__main__":
    main()
