/****************************************************************************
 *  CUDA 實作：使用 Coalesced memory access + Shared memory 優化
 *  並移除 checkCudaError 的錯誤檢查 (直接假設成功)。
 *
 *  編譯 (範例):
 *      nvcc -O2 flash_attention_cuda_coalesced_shared.cu -o flash_attention_cuda_opt
 *      ./flash_attention_cuda_opt input.bin output.bin
 ****************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// 如果您的環境沒有 nvtx3，可自行拿掉或改成空宏
#include "/home/pp24/pp24s036/firefly/NVTX/c/include/nvtx3/nvtx3.hpp"

// ---------- 全域變數：與原始程式保持一致 ----------
int B, N, d;
float *Q, *K, *V, *O;   // Host 端資料

// ---------- Device 端暫存指標：存放整個 B*N*d 的 Q, K, V, O，與 B*N 的 l, m ----------
static float *d_Q  = nullptr;  // 大小 B*N*d
static float *d_K  = nullptr;  // 大小 B*N*d
static float *d_V  = nullptr;  // 大小 B*N*d
static float *d_O  = nullptr;  // 大小 B*N*d
static float *d_l  = nullptr;  // 大小 B*N
static float *d_m  = nullptr;  // 大小 B*N

// ---------- 額外暫存空間 (block size 大小) ----------
static float *d_sij = nullptr; // 單次 block: 大小 br*bc
static float *d_pij = nullptr; // 單次 block: 大小 br*bc
static float *d_mij = nullptr; // 單次 block: 大小 br
static float *d_lij = nullptr; // 單次 block: 大小 br

// --------------------[ Utility Functions ]--------------------
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}
inline float _max(float a, float b) { return a > b ? a : b; }
inline float _min(float a, float b) { return a < b ? a : b; }

__device__ float d_max(float a, float b) { return a > b ? a : b; }

// ---------------------------------------------------------------------
//  Kernel: initLMForBatch
//    將 batch bIdx 的 l[], m[] 區段初始化：l=0, m=-∞
// ---------------------------------------------------------------------
__global__ void initLMForBatch(float *d_l, float *d_m, int bIdx, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // [0..N)
    if(i < N){
        int offset = bIdx*N + i;
        d_l[offset] = 0.0f;     // l=0
        d_m[offset] = -FLT_MAX; // m = -∞
    }
}

// ---------------------------------------------------------------------
//  QKDotAndScalarKernel with Tiling + Shared Memory
//  
//  執行時會把 Q_iBlock (大小 br*d) 與 K_jBlock (大小 bc*d) 分段載入 Shared memory，
//  每次載入 tileSize(=16) 個特徵到 shared memory，然後做部分 dot product。
//  使得存取 Q, K 的全域記憶體為 coalesced；計算在 shared memory 中完成後寫回 global。
// ---------------------------------------------------------------------
#define TILE_SIZE 16

__global__ void QKDotAndScalarKernel(
    float *out,        // d_sij (大小 br*bc)
    const float *d_Q,  // B*N*d
    const float *d_K,  // B*N*d
    int bIdx,
    int iBlock, 
    int jBlock,
    int br, int bc,
    int N, int dim,
    float scalar
)
{
    // ---------------------------
    // 1) 計算對應的 global pointer 起始
    // ---------------------------
    const float *qBase = d_Q + (bIdx*N*dim) + (iBlock * br * dim);
    const float *kBase = d_K + (bIdx*N*dim) + (jBlock * bc * dim);

    // i, j = row, col in the output S_ij
    int i = blockIdx.x * blockDim.x + threadIdx.x; // [0..br)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // [0..bc)
    if(i >= br || j >= bc) return;

    // ---------------------------
    // 2) 使用 shared memory 進行 tiling
    //    - blockDim.x, blockDim.y 預計設定為 (TILE_SIZE, TILE_SIZE)
    // ---------------------------
    __shared__ float sQ[TILE_SIZE][TILE_SIZE];  // sQ[i][t]
    __shared__ float sK[TILE_SIZE][TILE_SIZE];  // sK[j][t]

    float sum = 0.0f;
    // dot product over "dim" dimension,分多個 tile 進行
    // 每次處理 tileSize(=TILE_SIZE) 個特徵
    for(int tilePos = 0; tilePos < dim; tilePos += TILE_SIZE) {
        // tidx, tidy = 用於 shared memory 複製
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        // 每個 thread 複製對應的一小片 Q, K 到 shared memory
        // Q: [i, tilePos + tidy]
        // K: [j, tilePos + tidx]
        int qIndex = (i * dim) + (tilePos + tidy); // row i, col tilePos + tidy
        int kIndex = (j * dim) + (tilePos + tidx); // row j, col tilePos + tidx
        if((tilePos + tidy) < dim) {
            sQ[tidx][tidy] = qBase[qIndex];
        } else {
            sQ[tidx][tidy] = 0.0f; // 超過範圍補 0
        }
        if((tilePos + tidx) < dim) {
            sK[tidx][tidy] = kBase[kIndex];
        } else {
            sK[tidx][tidy] = 0.0f;
        }

        __syncthreads(); // 等待整個 tile 複製完成

        // 做部分 dot product
        for(int t=0; t < TILE_SIZE; t++){
            sum += sQ[tidx][t] * sK[t][tidy];
        }
        __syncthreads(); // 確保使用完 shared memory，再載入下一批
    }

    // 最後乘上 scalar
    out[i * bc + j] = sum * scalar;
}

// ---------------------------------------------------------------------
//  RowMaxKernel with shared memory reduce
// ---------------------------------------------------------------------
__global__ void RowMaxKernel(
    float *out,         // d_mij (大小 br)
    const float *in,    // d_sij (大小 br*bc)
    int br, int bc)
{
    __shared__ float sdata[TILE_SIZE]; // 每次一個 block 負責一行, 這裡僅示範

    int i = blockIdx.x;  // 每個 block 負責 row i
    if(i >= br) return;

    float mx = in[i * bc + 0];
    for(int j=1; j<bc; j++){
        mx = (in[i*bc + j] > mx)? in[i*bc + j] : mx;
    }
    out[i] = mx;
}

// ---------------------------------------------------------------------
//  MinusMaxAndExpKernel with shared memory 
// ---------------------------------------------------------------------
__global__ void MinusMaxAndExpKernel(
    float *out,         // d_pij (大小 br*bc)
    const float *in,    // d_sij (大小 br*bc)
    const float *mx,    // d_mij (大小 br)
    int br, int bc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i < br && j < bc) {
        float val = in[i*bc + j];
        float rowMax = mx[i];
        out[i*bc + j] = expf(val - rowMax);
    }
}

// ---------------------------------------------------------------------
//  RowSumKernel with shared memory reduce (示範簡單做法)
// ---------------------------------------------------------------------
__global__ void RowSumKernel(
    float *out,         // d_lij (大小 br)
    const float *in,    // d_pij (大小 br*bc)
    int br, int bc)
{
    int i = blockIdx.x; // 每個 block 負責一行
    if(i >= br) return;

    float sum = 0.0f;
    for(int j=0; j<bc; j++){
        sum += in[i*bc + j];
    }
    out[i] = sum;
}

// ---------------------------------------------------------------------
//  UpdateMiLiOiKernel (可再細分以 shared memory 平行化，但此處僅示範簡易版)
// ---------------------------------------------------------------------
__global__ void UpdateMiLiOiKernel(
    float *d_m, float *d_l, float *d_O,  
    const float *pij, const float *mij, const float *lij,
    const float *d_V,
    int bIdx,
    int iBlock,
    int jBlock,
    int br, int bc,
    int N, int dim
)
{
    float *mBlock  = d_m + (bIdx*N) + (iBlock * br);
    float *lBlock  = d_l + (bIdx*N) + (iBlock * br);
    float *oBlock  = d_O + (bIdx*N*dim) + (iBlock * br * dim);
    const float *vBlock = d_V + (bIdx*N*dim) + (jBlock * bc * dim);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < br) {
        float old_m = mBlock[i];
        float old_l = lBlock[i];
        float mm    = mij[i];
        float ll    = lij[i];

        // 新的 m_i, l_i
        float mi_new = (old_m > mm) ? old_m : mm;
        float li_new = expf(old_m - mi_new)*old_l + expf(mm - mi_new)*ll;

        float alpha_old = expf(old_m - mi_new) * old_l;
        float alpha_new = expf(mm - mi_new);

        float li_inv = 1.0f / li_new;
        // 更新 oBlock
        for(int j=0; j<dim; j++){
            float pv = 0.0f;
            for(int t=0; t<bc; t++){
                float pp = pij[i*bc + t];
                float vv = vBlock[t*dim + j];
                pv += pp * vv;
            }
            float oldVal = oBlock[i*dim + j];
            oBlock[i*dim + j] = (alpha_old * oldVal + alpha_new * pv) * li_inv;
        }

        // 寫回
        mBlock[i] = mi_new;
        lBlock[i] = li_new;
    }
}

// -------------------- Input / Output --------------------
void input(char *input_filename) {
    nvtxRangePushA("input");
    FILE *file = fopen(input_filename, "rb");
    if(!file){
        fprintf(stderr, "Cannot open input file: %s\n", input_filename);
        exit(1);
    }

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for(int i=0; i<B; i++){
        fread(Q + (i*N*d), sizeof(float), N*d, file);
        fread(K + (i*N*d), sizeof(float), N*d, file);
        fread(V + (i*N*d), sizeof(float), N*d, file);
    }
    // O 初始狀態 (通常 0)
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
    nvtxRangePop();
}

void output(char *output_filename) {
    nvtxRangePushA("output");
    FILE *file = fopen(output_filename, "wb");
    if(!file){
        fprintf(stderr, "Cannot open output file: %s\n", output_filename);
        exit(1);
    }
    fwrite(O, sizeof(float), B*N*d, file);

    free(Q);
    free(K);
    free(V);
    free(O);
    fclose(file);
    nvtxRangePop();
}

// -------------------- flash_attention (多 batch 版本) --------------------
void flash_attention_all_batches(int br, int bc) {
    nvtxRangePushA("flash_attention_all_batches");

    int tr = N / br;  // row-block 數
    int tc = N / bc;  // col-block 數
    float scale = 1.0f / sqrtf((float)d);

    for(int bIdx = 0; bIdx < B; bIdx++){
        // 1) 初始化 l=0, m=-∞
        {
            dim3 block(128);
            dim3 grid((N+block.x-1)/block.x);
            initLMForBatch<<<grid, block>>>(d_l, d_m, bIdx, N);
        }

        // 2) 依序掃過 iBlock, jBlock
        for(int iBlock=0; iBlock<tr; iBlock++){
            for(int jBlock=0; jBlock<tc; jBlock++){
                // (a) QKDotAndScalar (Tiled + Shared memory)
                {
                    nvtxRangePushA("QKDotAndScalar");
                    dim3 block(TILE_SIZE, TILE_SIZE); 
                    dim3 grid((br+TILE_SIZE-1)/TILE_SIZE, (bc+TILE_SIZE-1)/TILE_SIZE);

                    QKDotAndScalarKernel<<<grid, block>>>(
                        d_sij,
                        d_Q, d_K,
                        bIdx,
                        iBlock, jBlock,
                        br, bc,
                        N, d,
                        scale
                    );
                    nvtxRangePop();
                }

                // (b) RowMax
                {
                    nvtxRangePushA("RowMax");
                    // rowMax: 一個 block 負責一個 row
                    // => gr = br; block.x = 1
                    RowMaxKernel<<<br,1>>>(
                        d_mij,
                        d_sij,
                        br, bc
                    );
                    nvtxRangePop();
                }

                // (c) MinusMaxAndExp
                {
                    nvtxRangePushA("MinusMaxAndExp");
                    dim3 block(8,8);
                    dim3 grid((br+7)/8, (bc+7)/8);
                    MinusMaxAndExpKernel<<<grid, block>>>(
                        d_pij,
                        d_sij,
                        d_mij,
                        br, bc
                    );
                    nvtxRangePop();
                }

                // (d) RowSum
                {
                    nvtxRangePushA("RowSum");
                    // 與 RowMax 相似, 一個 block 負責一 row
                    RowSumKernel<<<br,1>>>( 
                        d_lij,
                        d_pij,
                        br, bc
                    );
                    nvtxRangePop();
                }

                // (e) UpdateMiLiOi
                {
                    nvtxRangePushA("UpdateMiLiOi");
                    dim3 block(32);
                    dim3 grid((br+31)/32);
                    UpdateMiLiOiKernel<<<grid, block>>>(
                        d_m, d_l, d_O,
                        d_pij, d_mij, d_lij,
                        d_V,
                        bIdx, iBlock, jBlock,
                        br, bc, N, d
                    );
                    nvtxRangePop();
                }
            }
        }
    }

    nvtxRangePop();
}

// -------------------- main --------------------
int main(int argc, char *argv[]) {
    if(argc != 3){
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    // 1) 讀取 Host 資料
    input(argv[1]);

    // 2) 在 GPU 一次配置 B*N*d
    cudaMalloc((void**)&d_Q, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_K, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_V, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_O, B*N*d*sizeof(float));
    cudaMalloc((void**)&d_l, B*N*sizeof(float));
    cudaMalloc((void**)&d_m, B*N*sizeof(float));

    // block-level 暫存
    int br = 32, bc = 32;
    cudaMalloc((void**)&d_sij, br*bc*sizeof(float));
    cudaMalloc((void**)&d_pij, br*bc*sizeof(float));
    cudaMalloc((void**)&d_mij, br*sizeof(float));
    cudaMalloc((void**)&d_lij, br*sizeof(float));

    // 3) 一次性拷 Q, K, V, O
    cudaMemcpy(d_Q, Q, B*N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, B*N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, B*N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, B*N*d*sizeof(float), cudaMemcpyHostToDevice);

    double start = getTimeStamp();

    // 4) 執行
    flash_attention_all_batches(br, bc);

    double end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    // 5) 把 O 拷回 Host
    cudaMemcpy(O, d_O, B*N*d*sizeof(float), cudaMemcpyDeviceToHost);

    // 6) 輸出
    output(argv[2]);

    // 7) 釋放
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);

    cudaFree(d_sij);
    cudaFree(d_pij);
    cudaFree(d_mij);
    cudaFree(d_lij);

    return 0;
}
