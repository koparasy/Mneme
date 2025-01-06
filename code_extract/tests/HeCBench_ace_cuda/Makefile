#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
VERIFY    = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main.out

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall --cuda-gpu-arch=$(ARCH) -pthread

# Linker Flags
LDFLAGS = -L/usr/lib/cuda/lib64 -lcudart_static -ldl -lrt

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY -fopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

# $(program): $(obj) Makefile
# 	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

$(program): $(source) reference.h Makefile
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100

