#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// #include "/home/pp24/pp24s036/firefly///nvtx/c/include///nvtx3///nvtx3.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

static const int WARP_SIZE = 32;

void input(char *input_filename);
void output(char *output_filename);
void self_attention(float *q, float *k, float *v, float *o);

void SoftMax(float *out, float *in);
void MulAttV(float *out, float *att, float *v);


__global__ void QKDotAndScalarKernel(float *d_out, const float *d_q, const float *d_k, float scalar, int N, int d);
__global__ void MulAttVKernel(float *d_out, const float *d_att, const float *d_v, int N, int d);
__global__ void SoftMaxKernel(float *d_att, int N);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    // 對每個批次執行自注意力
    for (int i = 0; i < B; i++) {
        self_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d)
        );
    }

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    //nvtxRangePushA("input");
    FILE *file = fopen(input_filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open input file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
    //nvtxRangePop();
}

void output(char *output_filename) {
    //nvtxRangePushA("output");
    FILE *file = fopen(output_filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open output file %s\n", output_filename);
        exit(EXIT_FAILURE);
    }

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
    //nvtxRangePop();
}

void self_attention(float *q, float *k, float *v, float *o) {
    //nvtxRangePushA("self_attention");

    // --------------- 1) 分配 GPU 端記憶體 ---------------
    float *d_q, *d_k, *d_att, *d_v, *d_out;
    size_t size_qk  = N * d * sizeof(float);
    size_t size_att = N * N * sizeof(float);
    size_t size_v   = N * d * sizeof(float);
    size_t size_o   = N * d * sizeof(float);
    
    //nvtxRangePushA("Allocate Device Memory");
    cudaMalloc((void **)&d_q,   size_qk);
    cudaMalloc((void **)&d_k,   size_qk);
    cudaMalloc((void **)&d_att, size_att);
    cudaMalloc((void **)&d_v,   size_v);
    cudaMalloc((void **)&d_out, size_o);
    //nvtxRangePop();
    // --------------- 2) Host -> Device 拷貝 ---------------
    //nvtxRangePushA("Copy Q, K, V to Device");
    cudaMemcpy(d_q, q, size_qk, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size_qk, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_v, cudaMemcpyHostToDevice);
    //nvtxRangePop();
    // --------------- 3) QKDotAndScalarKernel ---------------
    {
        //nvtxRangePushA("QKDotAndScalarKernel");
        float scalar = 1.0f / sqrtf((float)d);
        int threadsPerBlock = 1024; 
        int warpsPerBlock   = threadsPerBlock / WARP_SIZE;  // 1024/32 = 32
        int blocksPerGrid   = (N * N + warpsPerBlock - 1) / warpsPerBlock;

        QKDotAndScalarKernel<<<blocksPerGrid, threadsPerBlock>>>(d_att, d_q, d_k, scalar, N, d);
        //nvtxRangePop();
    }

    // --------------- 4) SoftMaxKernel (GPU 端) ---------------
    {
        //nvtxRangePushA("SoftMaxKernel");
        // 這裡假設 grid = N (一個block對應一個row)， blockDim.x=256
        dim3 grid(N);
        dim3 block(256);
        SoftMaxKernel<<<grid, block>>>(d_att, N);
        //nvtxRangePop();
    }

    // --------------- 5) MulAttVKernel ---------------
    {
        //nvtxRangePushA("MulAttVKernel");
        int threadsPerBlockMV = 256;
        int blocksPerGridMV   = (N * d + threadsPerBlockMV - 1) / threadsPerBlockMV;
        MulAttVKernel<<<blocksPerGridMV, threadsPerBlockMV>>>(d_out, d_att, d_v, N, d);
        //nvtxRangePop();
    }

    // --------------- 6) Device -> Host 拷貝 ---------------
    //nvtxRangePushA("Copy d_out to Device");
    cudaMemcpy(o, d_out, size_o, cudaMemcpyDeviceToHost);
    //nvtxRangePop();
    // --------------- 7) 釋放 GPU 記憶體 ---------------
    //nvtxRangePushA("Free Memory");
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_att);
    cudaFree(d_out);
    //nvtxRangePop();

    //nvtxRangePop();
}

// 每個 Block 有多少個 Warp
// 例如 blockDim.x = 1024，則 WARPS_PER_BLOCK = 1024 / 32 = 32
// 一個 block 會負責計算 32 個 out 索引
__global__ void QKDotAndScalarKernel(float *d_out, const float *d_q, const float *d_k,float scalar, int N, int d) {
    // warp-level reduce
    using WarpReduce = cub::WarpReduce<float>;

    // 每個 Block 有 WARPS_PER_BLOCK 個 Warp，
    // 所以要宣告對應數量的 TempStorage
    __shared__ typename WarpReduce::TempStorage warp_temp_storage[WARP_SIZE];

    // 確定 block 裡總共有多少 warp
    const int WARPS_PER_BLOCK = blockDim.x / WARP_SIZE;

    // block 的「全域 Warp ID」= (blockIdx.x * blockDim.x/32) + warp_id_in_block
    // => 表示第幾個 warp (從整個 grid 角度)
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int global_warp_id   = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    // lane_id: 該 thread 在它所在的 warp 裡的索引 (0~31)
    int lane_id = threadIdx.x % WARP_SIZE;

    // 總共有 N*N 個 output
    int total = N * N;

    // 如果這個 warp 超過要計算的總量，就直接 return
    if (global_warp_id >= total) return;

    // 決定此 warp 負責的 (i, j)
    int i = global_warp_id / N;
    int j = global_warp_id % N;

    // 每條 thread 分擔一部分 dot product
    // e.g. lane 0 負責 d_q[i*d + 0]~..., lane 1 負責 d_q[i*d + 1]~... 等
    float sum = 0.f;
    for (int t = lane_id; t < d; t += WARP_SIZE) {
        sum += d_q[i * d + t] * d_k[j * d + t];
    }

    // 進行 warp 級別的 reduce
    float warp_sum = WarpReduce(warp_temp_storage[warp_id_in_block]).Sum(sum);

    // lane 0 負責把結果寫回 global memory
    if (lane_id == 0) {
        d_out[global_warp_id] = warp_sum * scalar;
    }
}

__global__ void MulAttVKernel(float *d_out, const float *d_att, const float *d_v, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d;

    if (idx < total) {
        int i = idx / d; // 查詢位置
        int j = idx % d; // 維度位置
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            sum += d_att[i * N + t] * d_v[t * d + j];
        }
        d_out[i * d + j] = sum;
    }
}

__global__ void SoftMaxKernel(float *d_att, int N)
{
    // blockIdx.x = i (row index)
    // threadIdx.x 負責該 row 的若干元素
    // 假設 blockDim.x = 256 ，
    // 那若 N 大於 256，需要每條 thread 處理多個元素

    int i = blockIdx.x;  // 第 i row
    int tid = threadIdx.x;
    int stride = blockDim.x;  // 每次跳 stride

    // 1) 先在 shared memory 找到該 row 的 max 值
    __shared__ float row_max;  
    float local_max = -1e30f;  // 很小的初值
    // 先掃描自己負責的所有元素
    for (int j = tid; j < N; j += stride) {
        float val = d_att[i * N + j];
        if (val > local_max) {
            local_max = val;
        }
    }
    // 再做 block-level reduce (平均用 atomicMax 或者做平行 reduce)
    // 以下用 block reduce 的簡易寫法做示範：
    static __shared__ float sdata[256];  // 假設最多 blockDim.x=256
    sdata[tid] = local_max;
    __syncthreads();

    // 在 sdata 裡做 tree-reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    // 收斂之後 sdata[0] 就是該 row 的最大值
    if (tid == 0) row_max = sdata[0];
    __syncthreads();

    // 2) 計算 exp(...)，同時做總和
    __shared__ float row_sum;
    float local_sum = 0.f;
    for (int j = tid; j < N; j += stride) {
        // in-place 覆蓋: d_att = exp(d_att - row_max)
        float e = expf(d_att[i * N + j] - row_max);
        d_att[i * N + j] = e; // 暫存回去
        local_sum += e;
    }
    // 做 block reduce sum
    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) row_sum = sdata[0];
    __syncthreads();

    // 3) 將每個元素再除以 row_sum
    for (int j = tid; j < N; j += stride) {
        d_att[i * N + j] = d_att[i * N + j] / row_sum;
    }
}
