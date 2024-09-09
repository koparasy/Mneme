
ARCH=sm_70

all: example llvm

example:
	clang++  -gdwarf-4 -Wl,-rpath,../build/src/ -L../build/src/ --cuda-gpu-arch=$(ARCH) -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 example.cu -o example -lcuda -lcudart -lappreplay

llvm:
	clang++ -gdwarf-4 -Wl,-rpath,../build/src/ -L../build/src/ -O3 -save-temps -fpass-plugin=../build/pass/libRRPass.so --cuda-gpu-arch=$(ARCH) -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 example.cu -o example -lcuda -lcudart -lappreplay
	opt -O3 example-cuda-nvptx64-nvidia-cuda-sm_70.bc -S -o device.ll
	opt -O3 example-host-powerpc64le-unknown-linux-gnu.bc -S -o host.ll



clean:
	rm -f example *.ll *.cubin *.s *.cui *.bc *.bin *.o *.fatbin *.json 
