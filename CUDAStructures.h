#define WARP_SIZE 32

// One reported match: the private key (256-bit scalar) and the hash160 that hit
// the bloom filter. The host turns these into hex + a base58 P2PKH address.
struct FoundResult {
    uint64_t scalar[4];
    uint8_t  hash160[20];
};

__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);


#define CUDA_CHECK(ans) do { cudaError_t err = ans; if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)






