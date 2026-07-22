
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <fstream>
#include <vector>
#include <algorithm>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAHash.cu"     // single-TU build: pull in the hash implementation (see Makefile)
#include "CUDAUtils.h"
#include "CUDAStructures.h"
#include "bloom.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

// Record one bloom-filter hit into the results buffer. Lock-free: each hit claims
// a unique slot via atomicAdd; the global counter keeps counting past capacity so
// the host can report how many (if any) were dropped.
__device__ __forceinline__ void record_found(
    FoundResult* __restrict__ results,
    unsigned int* __restrict__ count,
    unsigned int capacity,
    const uint64_t scalar[4],
    const uint8_t h20[20])
{
    unsigned int idx = atomicAdd(count, 1u);
    if (idx < capacity) {
        FoundResult* fr = &results[idx];
#pragma unroll
        for (int k = 0; k < 4; ++k)  fr->scalar[k]  = scalar[k];
#pragma unroll
        for (int k = 0; k < 20; ++k) fr->hash160[k] = h20[k];
    }
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    unsigned int* __restrict__ d_found_count,
    FoundResult* __restrict__ d_found_results,
    unsigned int found_capacity,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left,
    const unsigned char* __restrict__ d_bloom
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane = (unsigned)(threadIdx.x & (WARP_SIZE - 1));

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];   
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        {
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            H160 h20 = getHash160_33_from_limbs(prefix, u256_of(x1));
            ++local_hashes; MAYBE_WARP_FLUSH();

            if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20.w)) {
                record_found(d_found_results, d_found_count, found_capacity, S, (const uint8_t*)h20.w);
            }

            H160 h20u = getHash160_65_from_limbs(u256_of(x1), u256_of(y1));
            if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20u.w)) {
                record_found(d_found_results, d_found_count, found_capacity, S, (const uint8_t*)h20u.w);
            }
        }

        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        for (int i = 0; i < half - 1; ++i) {
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4], y3[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);     
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3); 
                _ModMult(s, s, lam);
                ModSub256(y3, s, y1);   // full Y of this point

                uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                uint64_t addv=(uint64_t)(i+1);
                for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }

                H160 h20 = getHash160_33_from_limbs((y3[0]&1ULL)?0x03:0x02, u256_of(px3));
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20.w)) {
                    record_found(d_found_results, d_found_count, found_capacity, fs, (const uint8_t*)h20.w);
                }

                H160 h20u = getHash160_65_from_limbs(u256_of(px3), u256_of(y3));
                if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20u.w)) {
                    record_found(d_found_results, d_found_count, found_capacity, fs, (const uint8_t*)h20u.w);
                }
            }

            {
                uint64_t px3[4], s[4], lam[4], y3[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i); 

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                ModSub256(y3, s, y1);   // full Y of this point

                uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                uint64_t sub=(uint64_t)(i+1);
                for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }

                H160 h20 = getHash160_33_from_limbs((y3[0]&1ULL)?0x03:0x02, u256_of(px3));
                ++local_hashes; MAYBE_WARP_FLUSH();
                if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20.w)) {
                    record_found(d_found_results, d_found_count, found_capacity, fs, (const uint8_t*)h20.w);
                }

                H160 h20u = getHash160_65_from_limbs(u256_of(px3), u256_of(y3));
                if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20u.w)) {
                    record_found(d_found_results, d_found_count, found_capacity, fs, (const uint8_t*)h20u.w);
                }
            }

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4], y3[4];
            uint64_t px_i[4], py_i[4];
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            ModSub256(y3, s, y1);   // full Y of this point

            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
            uint64_t sub=(uint64_t)half;
            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }

            H160 h20 = getHash160_33_from_limbs((y3[0]&1ULL)?0x03:0x02, u256_of(px3));
            ++local_hashes; MAYBE_WARP_FLUSH();
            if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20.w)) {
                record_found(d_found_results, d_found_count, found_capacity, fs, (const uint8_t*)h20.w);
            }

            H160 h20u = getHash160_65_from_limbs(u256_of(px3), u256_of(y3));
            if (bloom_contains_hash160(d_bloom, (const uint8_t*)h20u.w)) {
                record_found(d_found_results, d_found_count, found_capacity, fs, (const uint8_t*)h20u.w);
            }

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        {
            uint64_t addv=(uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

// Load a Brainflayer-style *.blf bloom filter (2^32 bits = 512 MiB) into device
// global memory. On success *d_bloom_out holds the device pointer (caller frees).
static bool load_blf_to_device(const std::string& path, unsigned char** d_bloom_out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Error: cannot open bloom file: " << path << "\n";
        return false;
    }
    const std::streamsize fsize = f.tellg();
    f.seekg(0, std::ios::beg);
    if (fsize != (std::streamsize)BLOOM_SIZE) {
        std::cerr << "Error: bloom file size is " << (long long)fsize
                  << " bytes, expected " << (long long)BLOOM_SIZE
                  << " (2^32 bits / 512 MiB).\n";
        return false;
    }

    unsigned char* d_bloom = nullptr;
    cudaError_t e = cudaMalloc(&d_bloom, (size_t)BLOOM_SIZE);
    if (e != cudaSuccess) {
        std::cerr << "cudaMalloc(d_bloom): " << cudaGetErrorString(e) << "\n";
        return false;
    }

    const size_t CHUNK = 8ull * 1024 * 1024;
    std::vector<char> buf(CHUNK);
    size_t offset = 0;
    while (offset < (size_t)BLOOM_SIZE) {
        const size_t n = std::min(CHUNK, (size_t)BLOOM_SIZE - offset);
        if (!f.read(buf.data(), (std::streamsize)n)) {
            std::cerr << "Error: failed reading bloom file (at offset " << offset << ").\n";
            cudaFree(d_bloom);
            return false;
        }
        e = cudaMemcpy(d_bloom + offset, buf.data(), n, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            std::cerr << "cudaMemcpy(d_bloom): " << cudaGetErrorString(e) << "\n";
            cudaFree(d_bloom);
            return false;
        }
        offset += n;
    }

    *d_bloom_out = d_bloom;
    return true;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, bloom_path;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if ((arg == "--bloom" || arg == "-b") && i + 1 < argc) bloom_path = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
    }

    if (range_hex.empty() || bloom_path.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> -b|--bloom <file.blf> [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);

    auto is_zero_256 = [](const uint64_t a[4])->bool { return (a[0]|a[1]|a[2]|a[3]) == 0ull; };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow; borrow = (a[i] < borrow) ? 1ull : 0ull; am1[i] = v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t and0=a[0]&am1[0], and1=a[1]&am1[1], and2=a[2]&am1[2], and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length (end - start + 1) must be a power of two.\n"; return EXIT_FAILURE;
    }
    uint64_t len_minus1[4];
    {   uint64_t borrow=1ull;
        for (int i=0;i<4;++i) {
            uint64_t v=range_len[i]-borrow; borrow=(range_len[i]<borrow)?1ull:0ull; len_minus1[i]=v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) len_minus1[k]=range_len[k]; break; }
        }
    }
    {   uint64_t and0 = range_start[0] & len_minus1[0];
        uint64_t and1 = range_start[1] & len_minus1[1];
        uint64_t and2 = range_start[2] & len_minus1[2];
        uint64_t and3 = range_start[3] & len_minus1[3];
        if ((and0|and1|and2|and3) != 0ull) {
            std::cerr << "Error: start must be aligned to the range length.\n"; return EXIT_FAILURE;
        }
    }

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    unsigned char* d_bloom = nullptr;
    if (!load_blf_to_device(bloom_path, &d_bloom)) {
        return EXIT_FAILURE;
    }

    int threadsPerBlock=256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock=prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock=32;

    const uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        std::cerr << "Error: range length must be divisible by batch size (" << runtime_points_batch_size << ").\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64) { std::cerr << "Error: total batches too large for u64.\n"; return EXIT_FAILURE; }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    auto pick_threads_total = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        uint64_t q = total_batches_u64;
        while (t >= (uint64_t)threadsPerBlock) {
            if ((q % t) == 0ull) return t;
            t -= (uint64_t)threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper) upper = total_batches_u64;
    if (userUpper         < upper) upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ull;
    divmod_256_by_u64(range_len, threadsTotal, per_thread_cnt, r_u64);
    if (r_u64 != 0ull) { std::cerr << "Internal error: range_len not divisible by threadsTotal.\n"; return EXIT_FAILURE; }
    {   uint64_t qq[4], rr=0ull;
        divmod_256_by_u64(per_thread_cnt, (uint64_t)runtime_points_batch_size, qq, rr);
        if (rr != 0ull) { std::cerr << "Internal error: per-thread count is not a multiple of batch size.\n"; return EXIT_FAILURE; }
    }

    uint64_t* h_counts256     = nullptr;
    uint64_t* h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }

    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;
    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc); 
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];

            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    const unsigned int FOUND_CAPACITY = 1u << 16; // max matches retained for printing

    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    unsigned int *d_found_count=nullptr; FoundResult *d_found_results=nullptr;
    unsigned long long *d_hashes_accum=nullptr; unsigned int *d_any_left=nullptr;

    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
    ck(cudaMalloc(&d_Px,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_Rx,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
    ck(cudaMalloc(&d_Ry,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
    ck(cudaMalloc(&d_counts256,    threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
    ck(cudaMalloc(&d_found_count,   sizeof(unsigned int)),                            "cudaMalloc(d_found_count)");
    ck(cudaMalloc(&d_found_results, (size_t)FOUND_CAPACITY * sizeof(FoundResult)),    "cudaMalloc(d_found_results)");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)),          "cudaMalloc(d_hashes_accum)");
    ck(cudaMalloc(&d_any_left,     sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
    ck(cudaMemcpy(d_counts256,     h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
    { unsigned int zero = 0u; unsigned long long zero64=0ull;
      ck(cudaMemcpy(d_found_count, &zero,   sizeof(unsigned int),       cudaMemcpyHostToDevice), "init found_count");
      ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum"); }

    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(), "scalarMulKernelBase launch");
    }

    {
        uint64_t* h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
        ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
        ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

        uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half); std::free(h_Gy_half);
    }
    {
        uint64_t* h_scalarB = nullptr;
        cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
        h_scalarB[0] = (uint64_t)B;

        uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
        ck(cudaMalloc(&d_Jx,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
        ck(cudaMalloc(&d_Jy,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
        ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

        scalarMulKernelBase<<<1, 1>>>(d_scalarB, d_Jx, d_Jy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

        cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
        cudaFreeHost(h_scalarB);
    }

    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << (int)(threadsTotal / (uint64_t)threadsPerBlock) << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << B << "\n";
    std::cout << std::left << std::setw(20) << "Batches/SM"        << " : " << runtime_batches_per_sm << "\n";
    std::cout << std::left << std::setw(20) << "Batches/launch"    << " : " << slices_per_launch << " (per thread)\n";
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << (uint64_t)threadsTotal << "\n\n";
    std::cout << "======== Phase-1: BruteForce ==========================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv, cudaFuncCachePreferL1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    // Copy any newly recorded matches off the device and print them. Multi-target
    // mode never stops on a hit, so this is called periodically and at the end.
    unsigned int printed = 0;
    auto drain_found = [&]() {
        unsigned int h_count = 0;
        ck(cudaMemcpy(&h_count, d_found_count, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read found_count");
        if (h_count <= printed) return;
        unsigned int upto = (h_count < FOUND_CAPACITY) ? h_count : FOUND_CAPACITY;
        if (printed < upto) {
            unsigned int n = upto - printed;
            std::vector<FoundResult> tmp(n);
            ck(cudaMemcpy(tmp.data(), d_found_results + printed, (size_t)n * sizeof(FoundResult), cudaMemcpyDeviceToHost), "read found_results");
            for (unsigned int i = 0; i < n; ++i) {
                char hx[41];
                for (int b = 0; b < 20; ++b) std::snprintf(hx + 2*b, 3, "%02x", tmp[i].hash160[b]);
                std::cout << "\n======== FOUND MATCH! =================================\n";
                std::cout << "Private Key   : " << formatHex256(tmp[i].scalar) << "\n";
                std::cout << "Hash160       : " << hx << "\n";
                std::cout << "Address       : " << hash160_to_p2pkh(tmp[i].hash160) << "\n";
                std::cout.flush();
            }
        }
        if (h_count > FOUND_CAPACITY && printed < FOUND_CAPACITY) {
            std::cerr << "\n[warn] result buffer full (" << FOUND_CAPACITY << "); "
                      << (h_count - FOUND_CAPACITY) << " further match(es) counted but not stored.\n";
        }
        printed = h_count;
    };

    bool stop_all = false;
    bool completed_all = false;
    while (!stop_all) {
        if (g_sigint) std::cerr << "\n[Ctrl+C] Interrupt received. Finishing current kernel slice and exiting...\n";

        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel), "zero d_any_left");

        kernel_point_add_and_check_oneinv<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars, d_counts256,
            threadsTotal,
            B,
            slices_per_launch,
            d_found_count, d_found_results, FOUND_CAPACITY,
            d_hashes_accum,
            d_any_left,
            d_bloom
        );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
        }

        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                long double total_keys_ld = ld_from_u256(range_len);
                long double prog = total_keys_ld > 0.0L ? ((long double)h_hashes / total_keys_ld) * 100.0L : 0.0L;
                if (prog > 100.0L) prog = 100.0L;
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                          << " Mkeys/s | Count: " << h_hashes
                          << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
                std::cout.flush();
                lastHashes = h_hashes; tLast = now;
                drain_found();
            }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cudaStreamSynchronize(streamKernel);
        drain_found();
        std::cout.flush();
        if (stop_all || g_sigint) break;

        unsigned int h_any = 0u;
        ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");

        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

        if (h_any == 0u) { completed_all = true; break; }
    }

    cudaDeviceSynchronize();
    drain_found();
    std::cout << "\n";

    unsigned int h_total_found = 0;
    ck(cudaMemcpy(&h_total_found, d_found_count, sizeof(unsigned int), cudaMemcpyDeviceToHost), "final read found_count");

    int exit_code = EXIT_SUCCESS;

    if (g_sigint) {
        std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
        std::cout << "Search interrupted by user. Partial progress above.\n";
        exit_code = 130;
    } else if (completed_all) {
        std::cout << "======== DONE (exhaustive) ============================\n";
    } else {
        std::cout << "======== TERMINATED ===================================\n";
    }
    std::cout << "Total matches  : " << h_total_found << "\n";

    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_Rx); cudaFree(d_Ry);
    cudaFree(d_counts256); cudaFree(d_found_count); cudaFree(d_found_results); cudaFree(d_hashes_accum); cudaFree(d_any_left);
    cudaFree(d_bloom);
    cudaStreamDestroy(streamKernel);

    if (h_start_scalars) cudaFreeHost(h_start_scalars);
    if (h_counts256)     cudaFreeHost(h_counts256);

    return exit_code;
}
