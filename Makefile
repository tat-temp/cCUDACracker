TARGET      := CUDACyclone
# Single-TU build: CUDAHash.cu is #included into CUDACyclone.cu (NOT compiled separately),
# so the whole device program is one translation unit and builds with rdc=false. This lets
# ptxas optimize the hash + kernel together and use an intra-module register ABI for the
# by-value getHash160 (see CUDAHash.cuh for the measured effect). Listing CUDAHash.cu in SRC
# would emit a duplicate getHash160 device symbol at link.
SRC         := CUDACyclone.cu
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
GENCODE    := $(foreach arch,$(sort $(SM_ARCHS)),-gencode arch=compute_$(arch),code=sm_$(arch))

# rdc=false single-TU build: there are no cross-TU device calls, so -rdc=true and its device
# runtime (-lcudadevrt) are gone. --maxrregcount is likewise dropped: the kernel's
# __launch_bounds__(256,2) already caps registers, and without device linking there is no
# nvlink regcount error to work around. Dropping -rdc lets ptxas optimize hash+kernel together.
NVCC_FLAGS := -O3 -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

LDFLAGS    := -cudart=static

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Header + #included-TU dependencies, so editing any of them rebuilds CUDACyclone.o.
# CUDAHash.cu is #included (single-TU), so it is a prerequisite here, not a separate object;
# likewise the RCKangaroo field backend headers pulled in via CUDAMath.h -> ec_backend.cuh.
CUDACyclone.o: CUDACyclone.cu CUDAMath.h ec_backend.cuh third_party/RCKangaroo/RCGpuUtils.h \
               sha256.h CUDAHash.cuh CUDAHash.cu CUDAUtils.h CUDAStructures.h bloom.h

# Standalone CPU utility (Brainflayer's hex2blf): builds a .blf bloom filter from
# a text file of hash160s (one 40-hex-char hash160 per line). Not part of `all`.
hex2blf: hex2blf.c hex.h hash160.h bloom.h
	g++ -O2 hex2blf.c -o hex2blf -lm

clean:
	rm -f $(TARGET) $(OBJ) hex2blf
