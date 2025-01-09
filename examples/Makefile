
ARCH=sm_70

all: example llvm

example:
		clang++  -gdwarf-4 -Wl,-rpath,../build_lassen/src/ -L../build_lassen/src/ --cuda-gpu-arch=$(ARCH) -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 vec_add.cu -o example -lcuda -lcudart

llvm:
	clang++ -gdwarf-4 -Wl,-rpath,../build_lassen/src/ -L../build_lassen/src/ -O3 -save-temps -fpass-plugin=../build_lassen/pass/libRRPass.so --cuda-gpu-arch=$(ARCH) -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 vec_add.cu -o example -lcuda -lcudart



clean:
	rm -f example *.ll *.cubin *.s *.cui *.bc *.bin *.o *.fatbin *.json 
