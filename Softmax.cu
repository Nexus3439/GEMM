#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <ctime>
#include <iostream>

std::vector<float> cpu_softmax(const std::vector<float>& input, int r, int c) {
    if (input.empty()) return {};


    std::vector<float> result(input.size());
    for(int i = 0; i <r; ++i)
    {
        float max_val = *std::max_element(input.begin() + i * c, input.begin() + (i + 1) * c);
        std::vector<float> output(c);
        float sum = 0.0;
        for (int j = 0; j < c; ++j) {
            output[j] = std::exp(input[i * c + j] - max_val);
            sum += output[j];
        }
        for (int j = 0; j < c; ++j) {
            output[j] /= sum;
			result[i * c + j] = output[j];
        }
	}

	return result;
}

__global__ void Softmax(int input_rows, int input_cols, const float* input, float* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input_cols ||
        y >= input_rows)
    {
        return;
    }

    int output_idx = y * input_cols + x;
    float max_input = float(-1e27);
    // TODO: use shared memory to avoid every threads executing this part of code
    for (int i = 0; i < input_cols; ++i)
    {
        float c = input[y * input_cols + i];
        max_input = max_input < c ? c : max_input;
    }

    float cur = exp(input[output_idx] - max_input);
    float sum = float(0);
    for (int i = 0; i < input_cols; ++i)
    {
        sum += exp(input[y * input_cols + i] - max_input);
    }

    output[output_idx] = cur / sum;
}

__global__ void Softmax_opt(int input_rows, int input_cols, const float* input, float* output)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
	int r = id / input_cols;

    if (id >= input_cols * input_rows)
    {
        return;
    }

    float max_input = float(-1e27);
    float sum = 0;
    for (int i = 0; i < input_cols; ++i)
    {
        float c = input[r * input_cols + i];
        if(c > max_input)
        {
            sum *= exp(max_input - c);
            max_input = c;
		}

        sum += exp(c - max_input);
    }

    output[id] = exp(input[id] - max_input) / sum;
}

__global__ void Softmax_opt1(int input_rows, int input_cols, const float* input, float* output)
{
    __shared__ float mem[256];

    int tid = threadIdx.x;
    int r = blockIdx.x;

    float local_max = float(-1e27);
    float local_sum = 0;
    for (int i = tid; i < input_cols; i += blockDim.x)
    {
        float c = input[r * input_cols + i];
        if (c > local_max)
        {
            local_sum *= exp(local_max - c);
            local_max = c;
        }

        local_sum += exp(c - local_max);
    }

    mem[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset >= 1; offset /= 2)
    {
        if (tid < offset)
        {
			mem[tid] = max(mem[tid], mem[tid + offset]);
        }
        __syncthreads();
    }
    __syncthreads();

	float global_max = mem[0];
    __syncthreads();

	mem[tid] = local_sum * exp(local_max - global_max);

    for (int offset = blockDim.x / 2; offset >= 1; offset /= 2)
    {
        if (tid < offset)
        {
			mem[tid] += mem[tid + offset];
        }
        __syncthreads();
    }
    __syncthreads();

	float global_sum = mem[0];

    for (int i = tid; i < input_cols; i += blockDim.x)
    {
        output[r * input_cols + i] = exp(input[r * input_cols + i] - global_max) / global_sum;
    }
}

__global__ void Softmax_opt2(int input_rows, int input_cols, const float* input, float* output)
{
    __shared__ float mem[256/32];

    int tid = threadIdx.x;
    int r = blockIdx.x;

    float local_max = float(-1e27);
    float local_sum = 0;
    for (int i = tid; i < input_cols; i += blockDim.x)
    {
        float c = input[r * input_cols + i];
        if (c > local_max)
        {
            local_sum *= exp(local_max - c);
            local_max = c;
        }

        local_sum += exp(c - local_max);
    }

    float val = local_max;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (tid % 32 == 0)
    {
        mem[tid / 32] = val;
    }
    __syncthreads();

    if(tid < 8)
    {
        unsigned int mask = __activemask();
        val = mem[tid];
        for (int offset = 4; offset > 0; offset /= 2)
        {
            val = max(val, __shfl_down_sync(mask, val, offset));
        }

        if (tid == 0)
        {
            mem[0] = val;
		}
	}

    __syncthreads();

    float global_max = mem[0];

	local_sum *= exp(local_max - global_max);


    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (tid % 32 == 0)
    {
        mem[tid / 32] = local_sum;
    }
    __syncthreads();

    if (tid < 8)
    {
        unsigned int mask = __activemask();
        local_sum = mem[tid];
        for (int offset = 4; offset > 0; offset /= 2)
        {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }

        if (tid == 0)
        {
            mem[0] = local_sum;
        }
    }
    __syncthreads();

    float global_sum = mem[0];

    for (int i = tid; i < input_cols; i += blockDim.x)
    {
        output[r * input_cols + i] = exp(input[r * input_cols + i] - global_max) / global_sum;
    }
}

__global__ void Softmax_opt3(int input_rows, int input_cols, const float* input, float* output)
{
    __shared__ float mem[1024 / 32];

    int tid = threadIdx.x;
    int r = blockIdx.x;

    float local_max = float(-1e27);
    float local_sum = 0;
    for (int i = tid; i < input_cols / 4; i += blockDim.x * 2)
    {
        float4 c1 = *(reinterpret_cast<const float4*>(&(input[r * input_cols + 4 * i])));
        float4 c2 = *(reinterpret_cast<const float4*>(&(input[r * input_cols + 4 * (i + blockDim.x)])));
		float lm1 = max(max(c1.x, c1.y), max(c1.z, c1.w));
        float lm2 = max(max(c2.x, c2.y), max(c2.z, c2.w));
		float lm = max(lm1, lm2);
        if (lm > local_max)
        {
            local_sum *= __expf(local_max - lm);
            local_max = lm;
        }

        local_sum += (__expf(c1.x - local_max) + __expf(c1.y - local_max) + __expf(c1.z - local_max) + __expf(c1.w - local_max));
        local_sum += (__expf(c2.x - local_max) + __expf(c2.y - local_max) + __expf(c2.z - local_max) + __expf(c2.w - local_max));
    }

    float val = local_max;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    if (tid % 32 == 0)
    {
        mem[tid / 32] = val;
    }
    __syncthreads();

    if (tid < 32)
    {
        unsigned int mask = __activemask();
        val = mem[tid];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val = max(val, __shfl_xor_sync(mask, val, offset));
        }

        mem[tid] = val;
    }

    __syncthreads();

    float global_max = mem[tid / 32];

    local_sum *= __expf(local_max - global_max);


#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    if (tid % 32 == 0)
    {
        mem[tid / 32] = local_sum;
    }
    __syncthreads();

    if (tid < 32)
    {
        unsigned int mask = __activemask();
        local_sum = mem[tid];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            local_sum += __shfl_xor_sync(mask, local_sum, offset);
        }

        mem[tid] = local_sum;
    }
    __syncthreads();

    float global_sum = mem[tid / 32];
	float inv_global_sum = 1.0f / global_sum;
    for (int i = tid; i < input_cols / 4; i += blockDim.x * 2)
    {
        float4 c1 = *(reinterpret_cast<const float4*>(&(input[r * input_cols + 4 * i])));
        float4 c2 = *(reinterpret_cast<const float4*>(&(input[r * input_cols + 4 * (i + blockDim.x)])));
        output[r * input_cols + 4 * i + 0] = __expf(c1.x - global_max) * inv_global_sum;
        output[r * input_cols + 4 * i + 1] = __expf(c1.y - global_max) * inv_global_sum;
        output[r * input_cols + 4 * i + 2] = __expf(c1.z - global_max) * inv_global_sum;
        output[r * input_cols + 4 * i + 3] = __expf(c1.w - global_max) * inv_global_sum;
        output[r * input_cols + 4 * (i + blockDim.x) + 0] = __expf(c2.x - global_max) * inv_global_sum;
        output[r * input_cols + 4 * (i + blockDim.x) + 1] = __expf(c2.y - global_max) * inv_global_sum;
        output[r * input_cols + 4 * (i + blockDim.x) + 2] = __expf(c2.z - global_max) * inv_global_sum;
        output[r * input_cols + 4 * (i + blockDim.x) + 3] = __expf(c2.w - global_max) * inv_global_sum;
    }
}

void random_init(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

int main() {
    int r = 1024;
    int c = 32768*8;

    float* input, * output;
	cudaMalloc(&input, r * c * sizeof(float));
	cudaMalloc(&output, r * c * sizeof(float));

    std::vector<float> h_input(r*c);
    random_init(h_input.data(), r * c);

    std::vector<float> h_output(r * c);
    std::clock_t c_start = std::clock();
	h_output = cpu_softmax(h_input, r, c);
    std::clock_t c_end = std::clock();

    double time_elapsed_s = static_cast<double>(c_end - c_start) / CLOCKS_PER_SEC;

    std::cout << "CPU time: " << time_elapsed_s * 1000.0 << " ms" << std::endl;

     cudaMemcpy(input, h_input.data(), r * c * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

#define METHOD 4
#if METHOD == 0
    dim3 block(16, 16);
    dim3 grid((c + 15) / 16, (r + 15) / 16);
	Softmax << <grid, block >> > (r, c, input, output);
#elif METHOD == 1
    dim3 block(256);
    dim3 grid((r * c + 255) / 256);
	Softmax_opt << <grid, 256 >> > (r, c, input, output);
#elif METHOD == 2
    dim3 block(256);
    dim3 grid(r);
	Softmax_opt1 << <grid, 256 >> > (r, c, input, output);
#elif METHOD == 3
    dim3 block(256);
    dim3 grid(r);
	Softmax_opt2 << <grid, 256 >> > (r, c, input, output);  
#elif METHOD == 4
    dim3 block(1024);
    dim3 grid(r);
	Softmax_opt3 << <grid, 1024 >> > (r, c, input, output);
#endif

    {
        float ms = 0.0;
        int n_iter = 20;
        for (int i = 0; i < n_iter; ++i) {
            cudaEventRecord(start);
#if 0
            Softmax << <grid, block >> > (r, c, input, output);
#elif METHOD == 1
            Softmax_opt << <grid, block >> > (r, c, input, output);
#elif METHOD == 2
            Softmax_opt1 << <grid, block >> > (r, c, input, output);
#elif METHOD == 3
            Softmax_opt2 << <grid, block >> > (r, c, input, output);
#elif METHOD == 4
			Softmax_opt3 << <grid, block >> > (r, c, input, output);
#endif

            cudaDeviceSynchronize();
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float this_ms;
            cudaEventElapsedTime(&this_ms, start, end);
            ms += this_ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        printf("Performance: %fms\n", ms / n_iter);
    }

    cudaMemcpy(h_input.data(), output, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<r * c; ++i)
    {
        if (fabs(h_input[i] - h_output[i]) > 1e-7)
        {
            printf("Wrong result at index %d: expected %f, got %f\n", i, h_output[i], h_input[i]);
            break;
        }
	}

    cudaFree(input);
    cudaFree(output);
}