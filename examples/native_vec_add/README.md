# Simple native VecAdd example that builds with Mneme support 

## Build example
The example uses Mneme as an external installation and assumes there is a valid Mneme installation in the system. To build the example on an AMD architecture issue:

```bash
mkdir build/
cd build/
cmake -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DRR_Dir=<path-to-record-replay> -DENABLE_HIP=On ../
```

for a NVIDIA one you only need to replace `-DENABLE_HIP` with `-DENABLE_CUDA=On`

## Running Vector addition 

To run the example and record it issue:
```
LD_PRELOAD=<path-to-mneme-installation>/lib64/librecord.so RR_DATA_DIR=./ RR_FILE=record_replay_test.json ./build/vecAdd 400
```

This will generate two pairs of snapshot files. Every pair represents the state of the memory before and after invocation of the kernel. 


## Replaying the recorded kernel:

To replay one of the recorded kernels you can do the following:

First check the names of the recorded kernels in the *.json* file:
```
jq < record_replay_test.json
```
and pick a kernel-name out of the dictionaries.

Then issue the following command:

```
<path-to-mneme-installation>/bin/replay --kernel-name <named-of-kernel> --record-replay-json record_replay_test.json
```

## Performing a grid search to identify optimal execution configurations

```
<path-to-mneme-installation>/bin/GridSearch.py --kernel-name <named-of-kernel> --record-replay-json record_replay_test.json
```

This command will perform a GridSearch to identify optimal execution configurations.
