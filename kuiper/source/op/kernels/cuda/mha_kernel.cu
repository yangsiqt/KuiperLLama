#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cfloat>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
#include <base/tick.h>
namespace kernel {

// ============================================================================
// Naive MHA (original three-phase implementation)
//   Phase 1: Q·K^T → write full score to HBM
//   Phase 2: Softmax over full score in HBM
//   Phase 3: Read score from HBM, weighted sum with V
// ============================================================================

constexpr static int thread_num = 256;

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

__global__ void naive_multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                                  float* score_ptr, float* output, float* key_cache,
                                                  float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                                  int32_t head_num, int32_t head_size,
                                                  int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  extern __shared__ float s_query_head[];
  float scale = 1.f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    s_query_head[i] = query_head[i];
  }
  __syncthreads();

  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;

  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    float score = 0.0f;
    for (int i = 0; i < head_size; i += 4) {
      float4 key_val = *reinterpret_cast<float4*>(key_head + i);
      float4 query_val = *reinterpret_cast<float4*>(s_query_head + i);
      score += key_val.x * query_val.x + key_val.y * query_val.y + key_val.z * query_val.z +
               key_val.w * query_val.w;
    }
    score *= scale;
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; t++) {
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

// ============================================================================
// FlashAttention-2 for single-query decode
//
// Reference: "FlashAttention-2: Faster Attention with Better Parallelism
//             and Work Partitioning" (Tri Dao, 2023)
//
// For decode (seq_len_q = 1), the algorithm simplifies to tiling over KV:
//
//   Initialize: O = 0,  ℓ = 0,  m = -inf
//   For each KV tile j:
//     S_j = q · K_j^T · scale            (tiled score computation)
//     m̃_j = max(S_j)                     (tile-local max)
//     m_new = max(m, m̃_j)                (global running max)
//     P̃_j = exp(S_j - m_new)             (stable softmax weights)
//     ℓ = exp(m - m_new) · ℓ + sum(P̃_j)  (running normalizer)
//     O = exp(m - m_new) · O + P̃_j · V_j (unnormalized accumulation)
//     m = m_new
//   Output: O = O / ℓ                    (final normalization)
//
// Key differences from naive:
//   - No score matrix materialized in HBM (O(1) extra memory vs O(N))
//   - Single pass over KV cache (fused QK^T + softmax + V accumulation)
//   - Online softmax via running max/sum (FlashAttention-2 style)
// ============================================================================

constexpr int FA_TILE_KV = 128;

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void flash_attention_kernel(
    int32_t pos, int32_t seq_len,
    float* __restrict__ query,
    float* __restrict__ output,
    float* __restrict__ key_cache,
    float* __restrict__ value_cache,
    int32_t kv_dim, int32_t kv_mul,
    int32_t head_num, int32_t head_size,
    int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) return;

  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  int num_warps = num_threads / 32;

  // Shared memory layout:
  //   s_query  [head_size]     - query vector (loaded once)
  //   s_output [head_size]     - running output accumulator (unnormalized)
  //   s_scores [FA_TILE_KV]    - attention weights for current tile
  //   s_reduce [num_warps]     - workspace for warp-level reductions
  extern __shared__ float smem[];
  float* s_query  = smem;
  float* s_output = smem + head_size;
  float* s_scores = smem + 2 * head_size;
  float* s_reduce = smem + 2 * head_size + FA_TILE_KV;

  // Broadcast variables for online softmax state
  __shared__ float s_running_max;
  __shared__ float s_running_sum;
  __shared__ float s_tile_max;
  __shared__ float s_rescale_old;

  float scale = 1.0f / sqrtf(float(head_size));
  float* query_head = query + head * head_size;
  int head_offset = (head / kv_mul) * head_size;
  float* output_head = output + head * head_size;

  // Load query into SRAM, zero-initialize output accumulator
  for (int d = tid; d < head_size; d += num_threads) {
    s_query[d] = query_head[d];
    s_output[d] = 0.0f;
  }
  if (tid == 0) {
    s_running_max = -FLT_MAX;
    s_running_sum = 0.0f;
  }
  __syncthreads();

  int total_kv = pos + 1;
  int num_tiles = (total_kv + FA_TILE_KV - 1) / FA_TILE_KV;

  for (int tile = 0; tile < num_tiles; tile++) {
    int tile_start = tile * FA_TILE_KV;
    int tile_end = min(tile_start + FA_TILE_KV, total_kv);
    int tile_len = tile_end - tile_start;

    // ---- Step 1: S_j = q · K_j^T · scale ----
    // Each thread computes one or more Q·K dot products
    for (int t = tid; t < tile_len; t += num_threads) {
      float* key_head = key_cache + layer_offset + (tile_start + t) * kv_dim + head_offset;
      float score = 0.0f;
      for (int d = 0; d < head_size; d += 4) {
        float4 k = *reinterpret_cast<float4*>(key_head + d);
        float4 q = *reinterpret_cast<float4*>(s_query + d);
        score += k.x * q.x + k.y * q.y + k.z * q.z + k.w * q.w;
      }
      s_scores[t] = score * scale;
    }
    __syncthreads();

    // ---- Step 2: m̃_j = max(S_j) ----
    // Block-level max reduction via warp shuffles
    float local_max = -FLT_MAX;
    for (int t = tid; t < tile_len; t += num_threads) {
      local_max = fmaxf(local_max, s_scores[t]);
    }
    local_max = warp_reduce_max(local_max);
    int lane = tid & 31;
    int wid = tid >> 5;
    if (lane == 0) s_reduce[wid] = local_max;
    __syncthreads();
    if (tid == 0) {
      float mx = -FLT_MAX;
      for (int w = 0; w < num_warps; w++) mx = fmaxf(mx, s_reduce[w]);
      // m_new = max(m, m̃_j)
      float new_max = fmaxf(s_running_max, mx);
      s_tile_max = new_max;
      // exp(m_old - m_new): rescale factor for previous accumulation
      s_rescale_old = expf(s_running_max - new_max);
      s_running_max = new_max;
    }
    __syncthreads();

    float new_max = s_tile_max;
    float rescale = s_rescale_old;

    // ---- Step 3: P̃_j = exp(S_j - m_new),  ℓ̃_j = sum(P̃_j) ----
    // Numerically stable: all exponents are <= 0
    float local_sum = 0.0f;
    for (int t = tid; t < tile_len; t += num_threads) {
      float p = expf(s_scores[t] - new_max);
      s_scores[t] = p;
      local_sum += p;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_reduce[wid] = local_sum;
    __syncthreads();
    if (tid == 0) {
      float tile_sum = 0.0f;
      for (int w = 0; w < num_warps; w++) tile_sum += s_reduce[w];
      // ℓ = exp(m_old - m_new) · ℓ + sum(P̃_j)
      s_running_sum = s_running_sum * rescale + tile_sum;
    }
    __syncthreads();

    // ---- Step 4: O = diag(rescale) · O + P̃_j × V_j ----
    // FA-2: accumulate unnormalized output, delay division to the end
    for (int d = tid; d < head_size; d += num_threads) {
      float acc = s_output[d] * rescale;
      for (int t = 0; t < tile_len; t++) {
        float* val_head = value_cache + layer_offset + (tile_start + t) * kv_dim + head_offset;
        acc += s_scores[t] * val_head[d];
      }
      s_output[d] = acc;
    }
    __syncthreads();
  }

  // ---- Step 5: O = O / ℓ (final normalization, FA-2 style) ----
  float inv_sum = (s_running_sum > 0.0f) ? (1.0f / s_running_sum) : 0.0f;
  for (int d = tid; d < head_size; d += num_threads) {
    output_head[d] = s_output[d] * inv_sum;
  }
}

// ============================================================================
// Host dispatch: select between naive MHA and Flash Attention
// ============================================================================

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());
  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());
  cudaStream_t stream = config->stream;

#ifdef USE_NAIVE_MHA
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  naive_multi_head_attention_kernel<<<head_num, thread_num, head_size * sizeof(float), stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
#else
  UNUSED(score_tensor);
  constexpr int fa_block_size = 128;
  int num_warps = fa_block_size / 32;
  int smem_size = (2 * head_size + FA_TILE_KV + num_warps) * sizeof(float);
  flash_attention_kernel<<<head_num, fa_block_size, smem_size, stream>>>(
      pos, seq_len, query, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
#endif
}

}  // namespace kernel
