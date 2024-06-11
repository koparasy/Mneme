
ARCH=sm_70

all: example llvm

example:
	clang  --cuda-gpu-arch=$(ARCH) -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 example.cu -o example -lcuda -lcudart

llvm:
	clang -save-temps -fpass-plugin=../build/pass/libRRPass.so --cuda-gpu-arch=$(ARCH) -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 example.cu -o example -lcuda -lcudart
	opt -O0 example-cuda-nvptx64-nvidia-cuda-sm_70.bc -S -o device.ll
	opt -O0 example-host-powerpc64le-unknown-linux-gnu.bc -S -o host.ll



clean:
	rm example device.ll host.ll
