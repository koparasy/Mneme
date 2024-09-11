# Record Replay Tool

A simple tool allowing recording the execution of a GPU (CUDA) kernel and replaying that kernel as an independent executable.

The tool operates in 3 phases. During compile time the user needs to apply a provided LLVM pass to instrument the code. The pass detects all device global variables
and device functions and stores this information with the respective LLVM-IR in the global device memory. The compilation generates a `record-able` executable. 

The second phase involves running the application executable with a desired input and using `LD_PRELOAD` to enable recording. When recording before invoking a device kernel 
the pre-loaded library stores device memory in persistent storage and associates the memory with the device kernel and an LLVM IR file. At the end of the recorded execution
the pre-load library generates a database in the form of a `JSON` file containing information regarding the LLVM-IR files and the snapshots of device memory. 

During the third and last phase the user can replay the execution of an kernel as a separate independent executable. Besides executing it the user can modify the LLVM IR file and
auto-tune parameters such as kernel launch-bounds or kernel runtime execution parameters (e.g. Kernel Block and Grid Dimensions).


## Installation

To install the library and all the associated tools please issue the following commands:

```bash
mkdir build
cd build
installDir="$(pwd)/install"
cmake .. \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CUDA_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CUDA_COMPILER=clang++ \
-DLLVM_INSTALL_DIR=$(dirname $(dirname $(which clang))) \
-DENABLE_CUDA=On \
-DENABLE_TESTS=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ../
make
make install
export RR_INSTALL_DIR=${installDir}
```


After installation there should be the following files in the `$RR_INSTALL_DIR` directory:

```
$RR_INSTALL_DIR
|-- bin
|   `-- replay
`-- lib
    |-- cmake
    |   `-- RR
    |       |-- RRConfig.cmake
    |       |-- RRConfigVersion.cmake
    |       |-- RRTargets-relwithdebinfo.cmake
    |       `-- RRTargets.cmake
    |-- libRRPass.so
    `-- librecord.so

```

The `$RR_INSTALL_DIR/bin/replay` is the standalone replay tool the `$RR_INSTALL_DIR/lib/libRRPass.so` is the LLVM instrumentation pass 
and the `$RR_INSTALL_DIR/lib/librecord.so` is the library required to be pre-loaded. The installation command also installs the `cmake` RR target to allow  
other cmake packages to find the LLVMPass by using `find_package(RR)`.


## Example uses

Please check the ([RAJA](./examples/raja_vec_add//README.md)) and the cuda ([Native](./examples/native_cuda_vec_add/README.md)) examples.

## Limitations and Known Issues

- RecordReplay requires all application libraries to use share-linkage (*.so*) as RecordReplay uses `LD_PRELOAD` to overwrite the behavior of cuda memory calls.
- RecordReplay currently supports only recording executions on the default stream. We are working on supporting multiple streams.
- RecordReplay only support CUDA and we are working on supporting HIP programs. RecordReplay is tested with cuda@11.6.
- RecordReplay requires a modern clang/LLVM installation and is being developed with LLVM@17.


## Environmental Variables Controlling Recording:

- `RR_SYMBOLS` An environmental variable specyfing which kernels need to be recorded. The variable sets the function name of the kernel (mangled). Multiple functions can be separated by comma.
- `RR_FILE` The JSON file to store the record database metadata. If not provided the default name is `record_replay.json`
- `RR_DATA_DIR` The directory to store recorded snapshots and other files. By default this is set to the current working directory.
- `RR_VA_SIZE` The size of the Virtual Address page reservation in GB. The default value is set to 15GB. It is beneficial to set this to the total memory consumption of the total device memory. 



## Use of replay to search for near optimal configuration parameters.


We provide a simple python script which internall invokers the `replay` tool to search for optimal kernel launch parameters. The script assumes that the kernel to be optimized is a flexible kernel. uses a strided loop to make
the kernel invariant to the actual kernel configuration as suggested [here](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).

The user can specify the following:

```
python scripts/GridSearch.py --replay-path <$RR_INSTALL_DIR/bin/replay> --json <path-to-record.json> -k <name-of-the-kernel-to-be-optimized> --blocks 32,64,128,256,512,1024 --grids 320,480,640
```

and the algorithm will search in the cartesian product of `blocks X grids` to identify optimal configurations in terms of execution time. The script in the end generates a .csv file that contains all tested configurations and their total time.



## Contributions

We welcome all kinds of contributions: new features, bug fixes, documentation edits; it's all great!

To contribute, make a pull request, with develop as the destination branch.


# Release

RecordReplay is released under Apache License (Version 2.0) with LLVM exceptions. For more details, please see the [LICENSE](./LICENSE)


## Citation

If you use this software, please cite it as below:

```bibtex
@inproceedings{parasyris2023scalable,
  title={Scalable Tuning of (OpenMP) GPU Applications via Kernel Record and Replay},
  author={Parasyris, Konstantinos and Georgakoudis, Giorgis and Rangel, Esteban and Laguna, Ignacio and Doerfert, Johannes},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1--14},
  year={2023}
}
```

