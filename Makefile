TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
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

