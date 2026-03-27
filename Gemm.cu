
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define __CUDA_NO_BFLOAT16_OPERATORS__
#include <cuda_bf16.h>
#include <mma.h>
#include <stdio.h>
#include <vector>

using namespace nvcuda;
using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

#define FETCH_FLOAT4(addr) (*(reinterpret_cast<const float4*>(addr)))
#define STORE_FLOAT4(addr, val) (*(reinterpret_cast<float4*>(addr)) = (val))

inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {0xa0, 128},
        {0xa1, 128},
        {0xa3, 128},
        {0xb0, 128},
        {0xc0, 128},
        {0xc1, 128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}

__device__ __forceinline__ 
uint32_t SMemPtr2Addr(void* ptr)
{
    uint32_t addr;
    asm ("{.reg .u64 addr64;\n"
        "cvta.to.shared.u64 addr64, %1;\n"
        "cvt.u32.u64 %0, addr64;}\n"
        :"=r"(addr) 
        :"l"(ptr)
    );

    return addr;
}

__device__ __forceinline__
float* SMemAddr2Ptr(uint32_t addr) {
    uint64_t ptr64;
    asm (
        "{ .reg .u64 u64addr;\n"
        "  cvt.u64.u32 u64addr, %1;\n"     
        "  cvta.shared.u64 %0, u64addr; }\n" 
        : "=l"(ptr64)
        : "r"(addr)
        );
    return reinterpret_cast<float*>(ptr64);
}

__device__ __forceinline__
void Ldgsts32(uint32_t smem_addr, const void* gmem_ptr)
{
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :
        :"r"(smem_addr), "l"(gmem_ptr)
	);
}

__device__ __forceinline
void Ldgsts128(uint32_t smem_addr, const void* gmem_ptr)
{
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        :"r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__
void LdgstsWait()
{
    asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__
void Ldg32(float& v, const void* addr, bool guard)
{
    asm volatile(
        "{ .reg .pred p;\n"
        " setp.ne.s32 p, %2, 0;\n"
        "@p ld.global.f32 %0, [%1];}\n"
        :"=f"(v)
        :"l"(addr), "r"((int)guard)
    );
}

__device__ __forceinline__
void Ldg32(float& v, const void* addr)
{
    asm volatile(
        "ld.global.f32 %0, [%1];\n"
        :"=f"(v)
        :"l"(addr)
    );
}

__device__ __forceinline__
void Ldg128(float& v0, float& v1, float& v2, float& v3, const void* addr)
{
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        :"=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
        :"l"(addr)
    );
}

__device__ __forceinline__
void Stg32(const float& v, void* addr)
{
    asm volatile(
        "st.global.f32 [%1], %0;\n"
        :
        :"f"(v), "l"(addr)
    );
}

__device__ __forceinline__
void Stg128(const float& v0, const float& v1, const float& v2, const float& v3, void* addr)
{
    asm volatile(
        "st.global.v4.f32 [%4], {%0, %1, %2, %3};\n"
        :
        :"f"(v0), "f"(v1), "f"(v2), "f"(v3), "l"(addr)
    );
}

__device__ __forceinline__
void Lds32(float& v, uint32_t addr)
{
    asm volatile(
        "ld.shared.f32 %0, [%1];\n"
        :"=f"(v)
        :"r"(addr)
    );
}

__device__ __forceinline__
void Lds128(float& v0, float& v1, float& v2, float& v3, uint32_t addr)
{
    asm volatile(
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        :"=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
        :"r"(addr)
    );
}

__device__ __forceinline__
void Sts32(const float& v, uint32_t addr)
{
    asm volatile(
        "st.shared.f32 [%1], %0;\n"
        :
        :"f"(v), "r"(addr)
    );
}

__device__ __forceinline__
void Sts128(const float& v0, const float& v1, const float&v2, const float& v3, uint32_t addr)
{
    asm volatile(
        "st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n"
        :
        :"f"(v0), "f"(v1), "f"(v2), "f"(v3), "r"(addr)
    );
}

__global__ void GemmBf16TensorShape128x128x8(const bf16* a, const bf16* b, int m, int n, int k, float* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < 128; ++k) {
        sum += __bfloat162float(a[row * 128 + k]) * __bfloat162float(b[k * 128 + col]);
    }
    c[row * 128 + col] = __float2bfloat16(sum);
}

/* warp structure
 *
 * 
 * 
 * 
 *    |  128  |
 * ---|-------|-------|
 * 64 | warp0 | warp1 |
 * --------------------
 *    | warp2 | warp3 |
 *    ----------------
 *    | warp4 | warp5 |
 *    -----------------
 *    | warp6 | warp7 |
 *    -----------------
 */

 /* 
  * ------------------------------------------------------------------------------------------------------------------
  * |  t0  |  t1  |  t2  |  t3  |  t4  |  t5  |  t6  |  t7  ||  t0  |  ...                                           |
  * |  t8  |  t9  |  t10 |  t11 |  t12 |  t13 |  t14 |  t15 ||  t8  |  ...                                           |
  * |  t16 |  t17 |  t18 |  t19 |  t20 |  t21 |  t22 |  t23 ||  t16 |  ...                                           |
  * |  t24 |  t25 |  t26 |  t27 |  t28 |  t29 |  t30 |  t31 ||  t24 |  ...                                           |
  * |-------------------------------------------------------||--------------------------------------------------------
  * |  t0  |  ...                                           ||  t0  |  ...                                           |
  * |  t8  |  ...                                           ||  t8  |  ...                                           |
  * |  t16 |  ...                                           ||  t16 |  ...                                           |
  * |  t24 |  ...                                           ||  t24 |  ...                                           |
  * ------------------------------------------------------------------------------------------------------------------
  */
template<int METHOD, bool VECTORIZATION, bool ASYNC_LOAD>
__global__ void GemmBf16Shape256x256x8(const bf16* a, const bf16* b, int m, int n, int k, float* c)
{
    constexpr int TILE_M = 256;
    constexpr int TILE_E = TILE_M + 4;
    constexpr int TILE_N = 256;
    constexpr int TILE_K = 8;
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    bf16* a_smem = reinterpret_cast<bf16*>(smem);  // 1024 * 8
    bf16* b_smem = reinterpret_cast<bf16*>(smem + 16 * 1024);

    int block_x = blockIdx.y;
    int block_y = blockIdx.x;
    int t_id = threadIdx.x;
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;

    int num_tiles = k / TILE_K;

    int a_ldg_addr_base;
    int b_ldg_addr_base;
    int c_stg_addr_base;

    uint32_t a_lds_addr_base;
    uint32_t b_lds_addr_base;
    uint32_t a_sts_addr_base;
    uint32_t b_sts_addr_base;

    if constexpr (METHOD == 0)
    {
        a_ldg_addr_base = (block_x * TILE_M) * k + warp_id * 32 * k + ((lane_id / 4) * 4) * k + (lane_id % 4) * 2;
        if constexpr (VECTORIZATION)
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id * 8;
        }
        else
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id * 2;
        }
        c_stg_addr_base = (block_x * TILE_M * 2) * n * 2 + (block_y * TILE_N * 2) + (warp_id / 2) * 64 * n * 2 + (warp_id % 2) * 128 + (lane_id / 8) * 8 * n * 2 + (lane_id % 8) * 8;

        a_lds_addr_base = SMemPtr2Addr(a_smem + (warp_id / 2) * 64 + (lane_id / 8) * 8);
        b_lds_addr_base = SMemPtr2Addr(b_smem + (warp_id % 2) * 128 + (lane_id % 8) * 8);
        a_sts_addr_base = SMemPtr2Addr(a_smem + warp_id * 32 + (lane_id / 4) * 4 + (lane_id % 4) * 2 * TILE_E);
        if constexpr (VECTORIZATION)
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id * 8);
        }
        else
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id * 2);
        }
    }
    else
    {
        a_ldg_addr_base = (block_x * TILE_M) * k + warp_id * 16 * k + ((lane_id / 8) * 4) * k + (lane_id % 8);
        if constexpr (VECTORIZATION)
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id * 4;
        }
        else
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id;
        }
        c_stg_addr_base = (block_x * TILE_M) * n + (block_y * TILE_N) + (warp_id / 2) * 32 * n * 2 + (warp_id % 2) * 64 + ((lane_id / 16) * 8 + (lane_id % 2) * 4) * n * 2 + ((lane_id % 16) / 2) * 4;

        a_lds_addr_base = SMemPtr2Addr(a_smem + (warp_id / 2) * 32 + (lane_id / 16) * 8 + (lane_id % 2) * 4);
        b_lds_addr_base = SMemPtr2Addr(b_smem + (warp_id % 2) * 64 + ((lane_id % 16) / 2) * 4);
        a_sts_addr_base = SMemPtr2Addr(a_smem + warp_id * 16 + (lane_id / 8) * 4 + (lane_id % 8) * TILE_E);
        if constexpr (VECTORIZATION)
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id * 4);
        }
        else
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id);
        }
    }

    bf162 a_ldg_reg[4];
    bf162 b_ldg_reg[4];

    bf162 a_frag[2][8] = { 0.f };
    bf162 b_frag[2][8] = { 0.f };
    bf16 c_frag[16][16] = { __float2bfloat16(0.f) };

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if constexpr (ASYNC_LOAD)
        {
            Ldgsts32(a_sts_addr_base + i * sizeof(float), a + a_ldg_addr_base + i * k);
        }
        else
        {
            Ldg32(a_ldg_reg[i], a + a_ldg_addr_base + i * k);
        }
    }
    if constexpr (!ASYNC_LOAD)
    {
        Sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr_base);
    }

    if constexpr (VECTORIZATION)
    {
        if constexpr (ASYNC_LOAD)
        {
            Ldgsts128(b_sts_addr_base, b + b_ldg_addr_base);
        }
        else
        {
            Ldg128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b + b_ldg_addr_base);
            Sts128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b_sts_addr_base);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if constexpr (ASYNC_LOAD)
            {
                Ldgsts32(b_sts_addr_base + i * 64 * sizeof(bf16), b + b_ldg_addr_base + i * 64);
            }
            else
            {
                Ldg32(b_ldg_reg[i], b + b_ldg_addr_base + i * 64);

                Sts32(b_ldg_reg[i], b_sts_addr_base + i * 64 * sizeof(bf16));
            }
        }
    }

    a_sts_addr_base += 0x2000;
    b_sts_addr_base += 0x1000;

    if constexpr (ASYNC_LOAD)
    {
        LdgstsWait();
    }
    __syncthreads();

    Lds128(a_frag[0][0], a_frag[0][1], a_frag[0][2], a_frag[0][3], a_lds_addr_base);
    Lds128(a_frag[0][4], a_frag[0][5], a_frag[0][6], a_frag[0][7], a_lds_addr_base + 32 * sizeof(bf16));
    Lds128(b_frag[0][0], b_frag[0][1], b_frag[0][2], b_frag[0][3], b_lds_addr_base);
    Lds128(b_frag[0][4], b_frag[0][5], b_frag[0][6], b_frag[0][7], b_lds_addr_base + 64 * sizeof(bf16));

    for (int tile = 1; tile < num_tiles; ++tile)
    {
#pragma unroll
        for (int frag = 0; frag < TILE_K; ++frag)
        {
            if (frag == 0)
            {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    if constexpr (ASYNC_LOAD)
                    {
                        Ldgsts32(a_sts_addr_base + i * sizeof(float), a + a_ldg_addr_base + i * k + tile * TILE_K);
                    }
                    else
                    {
                        Ldg32(a_ldg_reg[i], a + a_ldg_addr_base + i * k + tile * TILE_K);
                    }
                }

                if constexpr (VECTORIZATION)
                {
                    if constexpr (ASYNC_LOAD)
                    {
                        Ldgsts128(b_sts_addr_base, b + b_ldg_addr_base + tile * TILE_K * n);
                    }
                    else
                    {
                        Ldg128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b + b_ldg_addr_base + tile * TILE_K * n);
                    }
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        if constexpr (ASYNC_LOAD)
                        {
                            Ldgsts32(b_sts_addr_base + i * 64 * sizeof(bf16), b + b_ldg_addr_base + i * 64 + tile * TILE_K * n);
                        }
                        else
                        {
                            Ldg32(b_ldg_reg[i], b + b_ldg_addr_base + i * 64 + tile * TILE_K * n);
                        }
                    }
                }
            }

            if (frag == TILE_K - 1)
            {
                if constexpr (ASYNC_LOAD)
                {
                    LdgstsWait();
                }
                else
                {
                    Sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr_base);

                    if constexpr (VECTORIZATION)
                    {
                        Sts128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b_sts_addr_base);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            Sts32(b_ldg_reg[i], b_sts_addr_base + i * 64 * sizeof(bf16));
                        }
                    }
                }

                int sign = (tile % 2 ? 1 : -1);
                a_sts_addr_base -= 0x2000 * sign;
                b_sts_addr_base -= 0x1000 * sign;
                a_lds_addr_base += 0x2000 * sign;
                b_lds_addr_base += 0x1000 * sign;

                __syncthreads();
            }

            int frag_read_index = frag % 2;
            int frag_write_index = (frag + 1) % 2;
            int a_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_E * sizeof(bf162);
            int b_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_N * sizeof(bf162);

            Lds128(a_frag[frag_write_index][0], a_frag[frag_write_index][1], a_frag[frag_write_index][2], a_frag[frag_write_index][3], a_lds_addr_base + a_lds_addr_offset);
            Lds128(a_frag[frag_write_index][4], a_frag[frag_write_index][5], a_frag[frag_write_index][6], a_frag[frag_write_index][7], a_lds_addr_base + 32 * sizeof(bf16) + a_lds_addr_offset);
            Lds128(b_frag[frag_write_index][0], b_frag[frag_write_index][1], b_frag[frag_write_index][2], b_frag[frag_write_index][3], b_lds_addr_base + b_lds_addr_offset);
            Lds128(b_frag[frag_write_index][4], b_frag[frag_write_index][5], b_frag[frag_write_index][6], b_frag[frag_write_index][7], b_lds_addr_base + 64 * sizeof(bf16) + b_lds_addr_offset);

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    bf162 a_val = a_frag[frag_read_index] + i;
                    bf162 b_val = b_frag[frag_read_index] + j;
                   
                    c_frag[2 * i + 0][2 * j + 0] = __hadd(c_frag[2 * i + 0][2 * j + 0], __hmul(a_val.x, b_val.x));
                    c_frag[2 * i + 0][2 * j + 1] = __hadd(c_frag[2 * i + 0][2 * j + 1], __hmul(a_val.x, b_val.y));
                    c_frag[2 * i + 1][2 * j + 0] = __hadd(c_frag[2 * i + 1][2 * j + 0], __hmul(a_val.y, b_val.x));
                    c_frag[2 * i + 1][2 * j + 1] = __hadd(c_frag[2 * i + 1][2 * j + 1], __hmul(a_val.y, b_val.y));
                }
            }
        }
    }

    // process the last tile
#pragma unroll
    for (int frag = 0; frag < TILE_K; ++frag)
    {
        int frag_read_index = frag % 2;
        if (frag < TILE_K - 1)
        {
            int frag_write_index = (frag + 1) % 2;
            int a_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_E * sizeof(bf162);
            int b_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_N * sizeof(bf162);

            Lds128(a_frag[frag_write_index][0], a_frag[frag_write_index][1], a_frag[frag_write_index][2], a_frag[frag_write_index][3], a_lds_addr_base + a_lds_addr_offset);
            Lds128(a_frag[frag_write_index][4], a_frag[frag_write_index][5], a_frag[frag_write_index][6], a_frag[frag_write_index][7], a_lds_addr_base + 32 * sizeof(bf16) + a_lds_addr_offset);
            Lds128(b_frag[frag_write_index][0], b_frag[frag_write_index][1], b_frag[frag_write_index][2], b_frag[frag_write_index][3], b_lds_addr_base + b_lds_addr_offset);
            Lds128(b_frag[frag_write_index][4], b_frag[frag_write_index][5], b_frag[frag_write_index][6], b_frag[frag_write_index][7], b_lds_addr_base + 64 * sizeof(bf16) + b_lds_addr_offset);
        }

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                bf162 a_val = a_frag[frag_read_index] + i;
                bf162 b_val = b_frag[frag_read_index] + j;

                c_frag[2 * i + 0][2 * j + 0] = __hadd(c_frag[2 * i + 0][2 * j + 0], __hmul(a_val.x, b_val.x));
                c_frag[2 * i + 0][2 * j + 1] = __hadd(c_frag[2 * i + 0][2 * j + 1], __hmul(a_val.x, b_val.y));
                c_frag[2 * i + 1][2 * j + 0] = __hadd(c_frag[2 * i + 1][2 * j + 0], __hmul(a_val.y, b_val.x));
                c_frag[2 * i + 1][2 * j + 1] = __hadd(c_frag[2 * i + 1][2 * j + 1], __hmul(a_val.y, b_val.y));
            }
        }
    }

    // write out the result
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            Stg128(__bfloat162float(c_frag[i][4 * j]), __bfloat162float(c_frag[i][4 * j + 1]), __bfloat162float(c_frag[i][4 * j + 2]), __bfloat162float(c_frag[i][4 * j + 3]), c + c_stg_addr_base + (i % 8) * n * 2 + (i / 8) * 32 * n * 2 + (j / 2) * 64 + 4 * j);
        }
    }
}

// (128, 8) x (8, 128)
// 256 threads per block, 8 warps
// 8x8 per thread

/* warp structure
 * -----------------
 * | warp0 | warp1 |
 * -----------------
 * | warp2 | warp3 |
 * ----------------
 * | warp4 | warp5 |
 * -----------------
 * | warp6 | warp7 |
 * -----------------
 */

/* threads structure in warp
 * here we don't process a consecutive 8x8 block per thread, instead for avoiding bank-conflict, the solution as follow, 
 * we have 4 sub-blocks here, each sub-block process 16x32 elements, which means each thread only process 4x4 elements in a sub-block.
 * ------------------------------------------------------------------------------------------------------------------
 * |  t0  |  t1  |  t2  |  t3  |  t4  |  t5  |  t6  |  t7  ||  t0  |  ...                                           |
 * |  t8  |  t9  |  t10 |  t11 |  t12 |  t13 |  t14 |  t15 ||  t8  |  ...                                           |
 * |  t16 |  t17 |  t18 |  t19 |  t20 |  t21 |  t22 |  t23 ||  t16 |  ...                                           |
 * |  t24 |  t25 |  t26 |  t27 |  t28 |  t29 |  t30 |  t31 ||  t24 |  ...                                           |
 * |-------------------------------------------------------||--------------------------------------------------------
 * |  t0  |  ...                                           ||  t0  |  ...                                           |
 * |  t8  |  ...                                           ||  t8  |  ...                                           |
 * |  t16 |  ...                                           ||  t16 |  ...                                           |
 * |  t24 |  ...                                           ||  t24 |  ...                                           |
 * ------------------------------------------------------------------------------------------------------------------
 */
 // sub tile (128, 8) X (8, 128)
 // 256 / 32 = 8 warps
 // 1 thread for processing a tile with 8x8 shape

template<int METHOD, bool VECTORIZATION, bool ASYNC_LOAD>
__global__ void GemmF32Shape128x128x8(const float* a, const float* b, int m, int n, int k, float* c)
{
    constexpr int TILE_M = 128;
    constexpr int TILE_E = TILE_M + 4;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 8;
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float* a_smem = reinterpret_cast<float*>(smem);  // 1024 * 8
    float* b_smem = reinterpret_cast<float*>(smem + 16 * 1024);

    int block_x = blockIdx.y;
    int block_y = blockIdx.x;
    int t_id = threadIdx.x;
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;

    int num_tiles = k / TILE_K;

    int a_ldg_addr_base;
    int b_ldg_addr_base;
    int c_stg_addr_base;

    uint32_t a_lds_addr_base;
    uint32_t b_lds_addr_base;
    uint32_t a_sts_addr_base;
    uint32_t b_sts_addr_base;

    if constexpr (METHOD == 0)
    {
        a_ldg_addr_base = (block_x * TILE_M) * k + warp_id * 16 * k + ((lane_id / 8) * 4) * k + (lane_id % 8);
        if constexpr (VECTORIZATION)
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id * 4;
        }
        else
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id;
        }
        c_stg_addr_base = (block_x * TILE_M) * n + (block_y * TILE_N) + (warp_id / 2) * 32 * n + (warp_id % 2) * 64 + (lane_id / 8) * 4 * n + (lane_id % 8) * 4;

        a_lds_addr_base = SMemPtr2Addr(a_smem + (warp_id / 2) * 32 + (lane_id / 8) * 4);
        b_lds_addr_base = SMemPtr2Addr(b_smem + (warp_id % 2) * 64 + (lane_id % 8) * 4);
        a_sts_addr_base = SMemPtr2Addr(a_smem + warp_id * 16 + (lane_id / 8) * 4 + (lane_id % 8) * TILE_E);
        if constexpr (VECTORIZATION)
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id * 4);
        }
        else
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id);
        }
    }
    else
    {
        a_ldg_addr_base = (block_x * TILE_M) * k + warp_id * 16 * k + ((lane_id / 8) * 4) * k + (lane_id % 8);
        if constexpr (VECTORIZATION)
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id * 4;
        }
        else
        {
            b_ldg_addr_base = (block_y * TILE_N) + (warp_id * n) + lane_id;
        }
        c_stg_addr_base = (block_x * TILE_M) * n + (block_y * TILE_N) + (warp_id / 2) * 32 * n + (warp_id % 2) * 64 + ((lane_id / 16) * 8 + (lane_id % 2) * 4) * n + ((lane_id % 16) / 2) * 4;

        a_lds_addr_base = SMemPtr2Addr(a_smem + (warp_id / 2) * 32 + (lane_id / 16) * 8 + (lane_id % 2) * 4);
        b_lds_addr_base = SMemPtr2Addr(b_smem + (warp_id % 2) * 64 + ((lane_id % 16) / 2) * 4);
        a_sts_addr_base = SMemPtr2Addr(a_smem + warp_id * 16 + (lane_id / 8) * 4 + (lane_id % 8) * TILE_E);
        if constexpr (VECTORIZATION)
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id * 4);
        }
        else
        {
            b_sts_addr_base = SMemPtr2Addr(b_smem + warp_id * TILE_N + lane_id);
        }
    }

    float a_ldg_reg[4];
    float b_ldg_reg[4];

    float a_frag[2][8] = { 0.f };
    float b_frag[2][8] = { 0.f };
    float c_frag[8][8] = { 0.f };

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if constexpr (ASYNC_LOAD)
        {
			Ldgsts32(a_sts_addr_base + i * sizeof(float), a + a_ldg_addr_base + i * k);
        }
        else
        {
            Ldg32(a_ldg_reg[i], a + a_ldg_addr_base + i * k);
        }
    }
    if constexpr (!ASYNC_LOAD)
    {
        Sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr_base);
    }

    if constexpr (VECTORIZATION)
    {
        if constexpr (ASYNC_LOAD)
        {
            Ldgsts128(b_sts_addr_base, b + b_ldg_addr_base);
        }
        else
        {
            Ldg128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b + b_ldg_addr_base);
            Sts128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b_sts_addr_base);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if constexpr (ASYNC_LOAD)
            {
                Ldgsts32(b_sts_addr_base + i * 32 * sizeof(float), b + b_ldg_addr_base + i * 32);
            }
            else
            {
                Ldg32(b_ldg_reg[i], b + b_ldg_addr_base + i * 32);

                Sts32(b_ldg_reg[i], b_sts_addr_base + i * 32 * sizeof(float));
            }
        }
    }

    a_sts_addr_base += 0x2000;
    b_sts_addr_base += 0x1000;

    if constexpr (ASYNC_LOAD)
    {
        LdgstsWait();
    }
    __syncthreads();

    Lds128(a_frag[0][0], a_frag[0][1], a_frag[0][2], a_frag[0][3], a_lds_addr_base);
    Lds128(a_frag[0][4], a_frag[0][5], a_frag[0][6], a_frag[0][7], a_lds_addr_base + 16 * sizeof(float));
    Lds128(b_frag[0][0], b_frag[0][1], b_frag[0][2], b_frag[0][3], b_lds_addr_base);
    Lds128(b_frag[0][4], b_frag[0][5], b_frag[0][6], b_frag[0][7], b_lds_addr_base + 32 * sizeof(float));

    for (int tile = 1; tile < num_tiles; ++tile)
    {
#pragma unroll
        for (int frag = 0; frag < TILE_K; ++frag)
        {
            if (frag == 0)
            {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    if constexpr (ASYNC_LOAD)
                    {
						Ldgsts32(a_sts_addr_base + i * sizeof(float), a + a_ldg_addr_base + i * k + tile * TILE_K);
                    }
                    else
                    {
                        Ldg32(a_ldg_reg[i], a + a_ldg_addr_base + i * k + tile * TILE_K);
                    }
                }

                if constexpr (VECTORIZATION)
                {
                    if constexpr (ASYNC_LOAD)
                    {
                        Ldgsts128(b_sts_addr_base, b + b_ldg_addr_base + tile * TILE_K * n);
                    }
                    else
                    {
                        Ldg128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b + b_ldg_addr_base + tile * TILE_K * n);
                    }
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        if constexpr (ASYNC_LOAD)
                        {
                            Ldgsts32(b_sts_addr_base + i * 32 * sizeof(float), b + b_ldg_addr_base + i * 32 + tile * TILE_K * n);
                        }
                        else
                        {
                            Ldg32(b_ldg_reg[i], b + b_ldg_addr_base + i * 32 + tile * TILE_K * n);
                        }
                    }
                }
            }

            if (frag == TILE_K - 1)
            {
                if constexpr (ASYNC_LOAD)
                {
                    LdgstsWait();
                }
                else
                {
                    Sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr_base);

                    if constexpr (VECTORIZATION)
                    {
                        Sts128(b_ldg_reg[0], b_ldg_reg[1], b_ldg_reg[2], b_ldg_reg[3], b_sts_addr_base);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            Sts32(b_ldg_reg[i], b_sts_addr_base + i * 32 * sizeof(float));
                        }
                    }
                }

                int sign = (tile % 2 ? 1 : -1);
                a_sts_addr_base -= 0x2000 * sign;
                b_sts_addr_base -= 0x1000 * sign;
				a_lds_addr_base += 0x2000 * sign;
				b_lds_addr_base += 0x1000 * sign;

                __syncthreads();
            }

            int frag_read_index = frag % 2;
            int frag_write_index = (frag + 1) % 2;
            int a_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_E * sizeof(float);
            int b_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_N * sizeof(float);

            Lds128(a_frag[frag_write_index][0], a_frag[frag_write_index][1], a_frag[frag_write_index][2], a_frag[frag_write_index][3], a_lds_addr_base + a_lds_addr_offset);
            Lds128(a_frag[frag_write_index][4], a_frag[frag_write_index][5], a_frag[frag_write_index][6], a_frag[frag_write_index][7], a_lds_addr_base + 16 * sizeof(float) + a_lds_addr_offset);
            Lds128(b_frag[frag_write_index][0], b_frag[frag_write_index][1], b_frag[frag_write_index][2], b_frag[frag_write_index][3], b_lds_addr_base + b_lds_addr_offset);
            Lds128(b_frag[frag_write_index][4], b_frag[frag_write_index][5], b_frag[frag_write_index][6], b_frag[frag_write_index][7], b_lds_addr_base + 32 * sizeof(float) + b_lds_addr_offset);

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    c_frag[i][j] += a_frag[frag_read_index][i] * b_frag[frag_read_index][j];
                }
            }
        }
    }

    // process the last tile
#pragma unroll
    for (int frag = 0; frag < TILE_K; ++frag)
    {
        int frag_read_index = frag % 2;
        if (frag < TILE_K - 1)
        {
            int frag_write_index = (frag + 1) % 2;
            int a_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_E * sizeof(float);
            int b_lds_addr_offset = ((frag + 1) % TILE_K) * TILE_N * sizeof(float);

            Lds128(a_frag[frag_write_index][0], a_frag[frag_write_index][1], a_frag[frag_write_index][2], a_frag[frag_write_index][3], a_lds_addr_base + a_lds_addr_offset);
            Lds128(a_frag[frag_write_index][4], a_frag[frag_write_index][5], a_frag[frag_write_index][6], a_frag[frag_write_index][7], a_lds_addr_base + 16 * sizeof(float) + a_lds_addr_offset);
            Lds128(b_frag[frag_write_index][0], b_frag[frag_write_index][1], b_frag[frag_write_index][2], b_frag[frag_write_index][3], b_lds_addr_base + b_lds_addr_offset);
            Lds128(b_frag[frag_write_index][4], b_frag[frag_write_index][5], b_frag[frag_write_index][6], b_frag[frag_write_index][7], b_lds_addr_base + 32 * sizeof(float) + b_lds_addr_offset);
        }

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                c_frag[i][j] += a_frag[frag_read_index][i] * b_frag[frag_read_index][j];
            }
        }
    }

    // write out the result
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            Stg128(c_frag[i][4 * j], c_frag[i][4 * j + 1], c_frag[i][4 * j + 2], c_frag[i][4 * j + 3], c + c_stg_addr_base + (i % 4) * n + (i / 4) * 16 * n + j * 32);
        }
    }
}


__global__ void bf16_gemm(bf16* a, bf16* b, float* c)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.f);

    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

void random_init(bf16* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2bfloat16(float(rand()) / RAND_MAX);
    }
}

bool check(const bf16* A,
    const bf16* B,
    const float* C,
    int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.f;
            for (int p = 0; p < k; ++p) {
                sum += __bfloat162float(A[i * k + p]) * __bfloat162float(B[j + p * n]);
            }

            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }

    return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float* c, const bf16* a, const bf16* b, unsigned int size)
{
    bf16* dev_a = 0;
    bf16* dev_b = 0;
    float* dev_c = 0;
    cudaError_t cudaStatus;
    std::vector<float> c_ref;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(bf16));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(bf16));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(bf16), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(bf16), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    int m = 256;
    int n = 256;
    int k = 256;
	dim3 grid((n + 255) / 256, (m + 255) / 256);
    // Launch a kernel on the GPU with one thread for each element.
    GemmBf16Shape256x256x8<0, 0, 0> << <grid, 256 >> > (dev_a, dev_b, m, n, k, dev_c);

    cudaEvent_t start, end;
    cudaStatus = cudaEventCreate(&start);
    cudaStatus = cudaEventCreate(&end);

    cudaStatus = cudaEventRecord(start);

	int n_iter = 10;
    for (int i = 0; i < 10; ++i)
    {
        // Launch a kernel on the GPU with one thread for each element.
        GemmBf16Shape256x256x8<0, 0, 0> << <grid, 256 >> > (dev_a, dev_b, m, n, k, dev_c);
    }

    cudaStatus = cudaEventRecord(end);
    cudaStatus = cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    double workload = n_iter * double(m) * n * k * 2;
    double gflops = (double(workload) / 1e9) / (double(ms) / 1e3);
    printf("Performance: %fGFLOPS\n", gflops);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	c_ref.resize(m * n);
	memcpy(c_ref.data(), c, m * n * sizeof(float));
    bool chk = check(a, b, c, m, n, k);
    printf("Matrix_C check: %s\n", chk ? "OK" : "Failed");

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

int main()
{
    const int arraySize = 256 * 256;
    bf16* a = new bf16[arraySize];
    bf16* b = new bf16[arraySize];
    float* c = new float[arraySize];

    random_init(a, arraySize);
    random_init(b, arraySize);
    for (int i = 0; i < arraySize; ++i)
    {
        a[i] = __float2bfloat16(1);
        b[i] = __float2bfloat16(1);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major >= 7)
    {
        printf("tensor core is supported.\n");
    }

    if (prop.major >= 8)
    {
        printf("BF16 is supported.\n");
    }

    double theory_gflops = prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor) * (prop.clockRate / 1e6) * 2;
	printf("theory gflops: %fGFlops\n", theory_gflops);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    /*for (int i = 0; i < 16; ++i)
    {
        printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
            (c[i * 16 + 0]), (c[i * 16 + 1]), (c[i * 16 + 2]), (c[i * 16 + 3]),
            (c[i * 16 + 4]), (c[i * 16 + 5]), (c[i * 16 + 6]), (c[i * 16 + 7]),
            (c[i * 16 + 8]), (c[i * 16 + 9]), (c[i * 16 + 10]), (c[i * 16 + 11]),
            (c[i * 16 + 12]), (c[i * 16 + 13]), (c[i * 16 + 14]), (c[i * 16 + 15]));
    }*/

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
