TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89

# Hopper (compute_90) needs CUDA >= 12. Add it automatically on a capable toolkit,
# or force it on/off with: make SM90=1  /  make SM90=0
CUDA_MAJOR ?= $(shell nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -n1)
ifeq ($(SM90),1)
SM_ARCHS += 90
else ifeq ($(SM90),0)
# explicitly disabled
else ifeq ($(shell [ -n "$(CUDA_MAJOR)" ] && [ "$(CUDA_MAJOR)" -ge 12 ] 2>/dev/null && echo yes),yes)
SM_ARCHS += 90
endif

SM_ARCHS += $(GPU_ARCH)

# Drop any arch the installed toolkit can't compile (e.g. an auto-detected
# compute_120 GPU on a pre-12.8 toolkit). Skip filtering if the query is empty.
NVCC_SUPPORTED := $(shell nvcc --list-gpu-arch 2>/dev/null | sed 's/compute_//')
ifneq ($(strip $(NVCC_SUPPORTED)),)
SM_ARCHS := $(filter $(NVCC_SUPPORTED),$(SM_ARCHS))
endif

# Highest target arch (numeric max). Embed its PTX too, so the binary still runs
# via JIT on a GPU newer than the toolkit (e.g. an sm_120 card built with CUDA 12.4).
SM_TOP  := $(shell printf '%s\n' $(SM_ARCHS) | sort -n | tail -n1)
GENCODE := $(foreach arch,$(sort $(SM_ARCHS)),-gencode arch=compute_$(arch),code=sm_$(arch)) \
           -gencode arch=compute_$(SM_TOP),code=compute_$(SM_TOP)

# --maxrregcount=128 matches the kernel's __launch_bounds__(256,2) ceiling so that
# separately-compiled device functions (e.g. getHash160_65_from_limbs) stay within
# the caller's register budget; without it nvlink rejects the call on sm_90.
NVCC_FLAGS := -O3 -rdc=true -use_fast_math --maxrregcount=128 --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

LDFLAGS    := -lcudadevrt -cudart=static

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Header dependencies, so editing a header forces the affected object to rebuild
# (the pattern rule above supplies the recipe; these add prerequisites).
CUDACyclone.o: CUDACyclone.cu CUDAMath.h sha256.h CUDAHash.cuh CUDAUtils.h CUDAStructures.h bloom.h
CUDAHash.o:    CUDAHash.cu CUDAHash.cuh

# Standalone CPU utility (Brainflayer's hex2blf): builds a .blf bloom filter from
# a text file of hash160s (one 40-hex-char hash160 per line). Not part of `all`.
hex2blf: hex2blf.c hex.h hash160.h bloom.h
	g++ -O2 hex2blf.c -o hex2blf -lm

clean:
	rm -f $(TARGET) $(OBJ) hex2blf

