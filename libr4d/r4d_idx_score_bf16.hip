// The compressed indexer's group score (Hy4-flash, docs/OPS.md "idx_score"): how much one query
// wants each closed group of key history.
//
//   w_h      = gate[m][h] * (heads * n)^-0.5
//   score[m][b] = ( sum_{h < 8} w_h * relu(d_h) ) + ( sum_{h >= 8} w_h * relu(d_h) )
//   d_h      = q[m][h] . key(bt[seq(m)][b])
//
// heads 16 and n 128 are compiled in. One wave scores one query row against 16 keys: the 16 heads'
// query fragments are the A operand (loaded once a row), the 16 keys the B operand, read straight
// into registers in fragment order, and eight WMMA steps give D[head][key]. A lane then holds one
// key against eight heads (8 * half + e): relu, times w_h and the sum over e run in the lane, and
// one lane exchange adds the two halves, which is the order the reference states.
//
// Two forms run that same chain on the same fragments, so their bytes are equal (r4d_selftest and
// tests/r4d_emu compare them):
//  - per row (decode heights): a workgroup is four waves on 64 consecutive keys of one row, grid
//    (M, ceil(columns / 64)). Every (row, group) reads its 256-byte key and the row's query again.
//  - row-tiled (prefill, M >= kTileRows): a workgroup is eight waves on 16 rows x 128 keys, grid
//    (ceil(M / 16), ceil(columns / 128)). A wave keeps its 16 keys' fragments in registers across
//    the rows and reloads them only where the rows' sequence changes; the rows' queries are staged
//    in LDS eight at a time. About 48 bytes a (row, group) in place of 512 at a long prefill.
// In both a wave whose keys all lie past its row's `nc` stores -inf without loading or multiplying.
// Rows do not interact, so a row's bytes are independent of M, of the form and of the other rows.
//
// Dot products run through WMMA, so a score is not bit-equal to libref's f32 sum; the order of the
// head sums is. A key past `nc[seq]`, without a page, or with a page outside the cache scores
// -inf, as does every column past the table. The op does no causal work (qsa_select does): a
// causal skip would leave future columns unscored, which libref and the other backends score, so
// it needs the op's contract changed first (docs/OPS.md).
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <hip/hip_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kWave    = 32;
constexpr int kN       = 128;
constexpr int kHeads   = 16;
constexpr int kKS      = kN / 16;       /* WMMA K steps */
constexpr int kWaves   = 4;
constexpr int kKeysWg  = kWaves * 16;   /* keys a workgroup */
/* The row-tiled form. */
constexpr int kTRows   = 16;            /* rows a workgroup */
constexpr int kTStage  = 8;             /* rows staged in LDS at once */
constexpr int kTWaves  = 8;
constexpr int kTKeys   = kTWaves * 16;  /* keys a workgroup */
constexpr int kTQP     = kN + 8;        /* a head's LDS pitch (shorts): the 32 lanes' 8-byte reads
                                           fall on distinct banks; a multiple of 4 for fragP */
constexpr int kTileRows = 256;          /* the launcher's rule: tiled from this many rows */

int g_idx_score_form = 0;   /* r4d_idx_score_pin_form */

/* The row's sequence: cu is nondecreasing, so a binary search for the last cu[i] <= m; -1 when m
 * is in none. */
__device__ __forceinline__ int idx_seq_of(const int32_t* __restrict__ cu, int nseq, int m) {
    int lo = 0, hi = nseq;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cu[mid] <= m) lo = mid; else hi = mid;
    }
    return (m >= cu[lo] && m < cu[lo + 1]) ? lo : -1;
}

/* D[head 8 * half + e][key col] to the score: relu, the head weight and the in-lane sum, e
 * ascending, then the first half of the heads plus the second. Every lane must be active. */
__device__ __forceinline__ float idx_head_sum(v8f acc, const unsigned short* gp, int half,
                                              float wscale) {
    float s = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float w = bf16_to_f32(gp[e]) * wscale;
        const float d = acc[e];
        s = s + w * (d < 0.0f ? 0.0f : d);
    }
    const float other = swap16(s);
    return half == 0 ? s + other : other + s;
}

#define IDX_SCORE_PARAMS                                                                        \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ gate,             \
    const unsigned short* __restrict__ key, const int32_t* __restrict__ bt,                     \
    const int32_t* __restrict__ cu, const int32_t* __restrict__ nc, float* __restrict__ out,    \
    int nseq, int cols, int out_cols, int pages, long q_ld, long gate_ld, long key_ld,          \
    long bt_ld, long out_ld, float wscale
__device__ __forceinline__ void idx_score_body(IDX_SCORE_PARAMS) {

    const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x >> 5;
    const int half = lane >> 4, col = lane & 15;
    const int m  = (int)blockIdx.x;
    const int b0 = (int)blockIdx.y * kKeysWg + wave * 16;
    if (b0 >= out_cols) return;          /* wave-uniform; nothing below synchronises */

    const int seq = idx_seq_of(cu, nseq, m);
    const int nlive = seq < 0 ? 0 : (nc[seq] < cols ? nc[seq] : cols);
    const int b = b0 + col;
    if (b0 >= nlive) {                   /* wave-uniform: no key of this wave is live */
        if (half == 0 && b < out_cols) out[(size_t)m * out_ld + b] = -INFINITY;
        return;
    }

    /* The key of this lane's column: its page, or page 0 (always present) when it has none. */
    const int page = (b < nlive) ? bt[(size_t)seq * bt_ld + b] : -1;
    const bool ok  = page >= 0 && page < pages;

    /* A operand: head `col`'s query row. B operand: the key. Same fragment layout for both. */
    const unsigned short* qp = q + (size_t)m * q_ld + (size_t)col * kN + 4 * half;
    const unsigned short* kp = key + (size_t)(ok ? page : 0) * key_ld + 4 * half;
    v8bf qf[kKS], kf[kKS];
#pragma unroll
    for (int ks = 0; ks < kKS; ++ks) qf[ks] = fragP(qp + ks * 16);
#pragma unroll
    for (int ks = 0; ks < kKS; ++ks) kf[ks] = fragP(kp + ks * 16);

    v8f acc;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < kKS; ++ks) acc = mma(qf[ks], kf[ks], acc);

    const float total = idx_head_sum(acc, gate + (size_t)m * gate_ld + 8 * half, half, wscale);
    if (half == 0 && b < out_cols) out[(size_t)m * out_ld + b] = ok ? total : -INFINITY;
}

/* The row-tiled form. No wave returns early: all reach the staging barriers. */
__device__ __forceinline__ void idx_tile_body(IDX_SCORE_PARAMS, int M) {
    __shared__ unsigned short qs[kTStage * kHeads * kTQP];
    __shared__ int sh_seq[kTRows];

    const int t = threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;
    const int half = lane >> 4, col = lane & 15;
    const int m0 = (int)blockIdx.x * kTRows;
    const int b0 = (int)blockIdx.y * kTKeys + wave * 16, b = b0 + col;
    if (t < kTRows) sh_seq[t] = m0 + t < M ? idx_seq_of(cu, nseq, m0 + t) : -1;

    /* The keys of the sequence last loaded (kseq), wave-uniform. */
    int kseq = INT_MIN, nlive = 0;
    bool ok = false;
    v8bf kf[kKS];
    constexpr int kRowWords = kHeads * kN / 4;   /* 8-byte words a row's query */
    for (int r0 = 0; r0 < kTRows && m0 + r0 < M; r0 += kTStage) {
        __syncthreads();   /* the previous stage read; sh_seq written */
        for (int i = t; i < kTStage * kRowWords; i += kTWaves * kWave) {
            const int r = i / kRowWords, e = (i % kRowWords) * 4;
            if (m0 + r0 + r < M)
                *(uint2*)&qs[(r * kHeads + e / kN) * kTQP + e % kN] =
                    *(const uint2*)(q + (size_t)(m0 + r0 + r) * q_ld + e);
        }
        __syncthreads();
        if (b0 >= out_cols) continue;    /* wave-uniform */
        for (int r = 0; r < kTStage && m0 + r0 + r < M; ++r) {
            const int m = m0 + r0 + r, seq = sh_seq[r0 + r];
            if (seq != kseq) {
                kseq = seq;
                nlive = seq < 0 ? 0 : (nc[seq] < cols ? nc[seq] : cols);
                if (b0 < nlive) {
                    /* This lane's key: its page, or page 0 (always present) when it has none. */
                    const int page = b < nlive ? bt[(size_t)seq * bt_ld + b] : -1;
                    ok = page >= 0 && page < pages;
                    const unsigned short* kp = key + (size_t)(ok ? page : 0) * key_ld + 4 * half;
#pragma unroll
                    for (int ks = 0; ks < kKS; ++ks) kf[ks] = fragP(kp + ks * 16);
                }
            }
            float* const o = out + (size_t)m * out_ld + b;
            if (b0 >= nlive) {           /* wave-uniform */
                if (half == 0 && b < out_cols) *o = -INFINITY;
                continue;
            }
            const unsigned short* qp = qs + (r * kHeads + col) * kTQP + 4 * half;
            v8f acc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < kKS; ++ks) acc = mma(fragP(qp + ks * 16), kf[ks], acc);
            const float total =
                idx_head_sum(acc, gate + (size_t)m * gate_ld + 8 * half, half, wscale);
            if (half == 0 && b < out_cols) *o = ok ? total : -INFINITY;
        }
    }
}

template <class CH>
__global__ __launch_bounds__(kWaves * kWave)
void idx_score_kernel(IDX_SCORE_PARAMS) {
    if (CH::on) r4d_chain_wait();
    idx_score_body(q, gate, key, bt, cu, nc, out, nseq, cols, out_cols, pages, q_ld, gate_ld,
                   key_ld, bt_ld, out_ld, wscale);
    if (CH::on) r4d_chain_done();
}

template <class CH>
__global__ __launch_bounds__(kTWaves * kWave)
void idx_score_tile_kernel(IDX_SCORE_PARAMS, int M) {
    if (CH::on) r4d_chain_wait();
    idx_tile_body(q, gate, key, bt, cu, nc, out, nseq, cols, out_cols, pages, q_ld, gate_ld,
                  key_ld, bt_ld, out_ld, wscale, M);
    if (CH::on) r4d_chain_done();
}
#undef IDX_SCORE_PARAMS

}  // namespace

// ============================================================== the plugin ABI entry point

// Test hook (r4d.h): 0 the launcher's rule, 1 the per-row form, 2 the row-tiled form.
extern "C" void r4d_idx_score_pin_form(int form) { g_idx_score_form = form; }

/* row_part: the step's cu rebased to the part's first row, which the part then scores against and
 * qsa_select reads over the part's rows; rows past the step fall in no sequence and score -inf. */
__global__ void idx_rebase_kernel(const int32_t* cu, int32_t* out, int n, int row0) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = cu[i] - row0;
}

enum { IS_Q = 0, IS_GATE = 1, IS_K = 2, IS_BT = 3, IS_CU = 4, IS_NC = 5, IS_SCORE = 6,
       IS_CU_PART = 7 };

extern "C" int r4d_rad_idx_score(const RadArgs* a, RadStream stream) {
    const RadTensor* q  = r4d_opt(a, IS_Q);
    const RadTensor* g  = r4d_opt(a, IS_GATE);
    const RadTensor* k  = r4d_opt(a, IS_K);
    const RadTensor* bt = r4d_opt(a, IS_BT);
    const RadTensor* cu = r4d_opt(a, IS_CU);
    const RadTensor* nc = r4d_opt(a, IS_NC);
    const RadTensor* o  = r4d_opt(a, IS_SCORE);
    if (!q || !g || !k || !bt || !cu || !nc || !o) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0), H = r4d_geti_or(a, "heads", 0);
    if (n != kN || H != kHeads) return R4D_NO(RAD_E_SHAPE);
    if (q->dtype != RAD_BF16 || g->dtype != RAD_BF16 || k->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (bt->dtype != RAD_I32 || cu->dtype != RAD_I32 || nc->dtype != RAD_I32 ||
        o->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    /* q: [M, heads, n] with heads and n dense, or [M, heads * n]. */
    long long M = 0;
    if (q->rank == 3) {
        if (q->shape[1] != H || q->shape[2] != n || q->stride[2] != 1 || q->stride[1] != n)
            return R4D_NO(RAD_E_SHAPE);
        M = q->shape[0];
    } else if (q->rank == 2) {
        if (q->shape[1] != H * n || q->stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        M = q->shape[0];
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (M < 1 || M > (1ll << 30)) return R4D_NO(RAD_E_SHAPE);
    if (g->rank != 2 || g->shape[0] != M || g->shape[1] < H || g->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    /* row_part p (dsa_score's split): q and gate span the step, score holds R = its rows, rows
     * [p R, p R + R) are scored against cu - p R, written to cu_part first. */
    const long long part = r4d_geti_or(a, "row_part", -1);
    const RadTensor* cp = r4d_opt(a, IS_CU_PART);
    long long row0 = 0;
    if (part >= 0) {
        if (!cp || cp->dtype != RAD_I32) return R4D_NO(cp ? RAD_E_DTYPE : RAD_E_INVAL);
        if (o->rank != 2 || o->shape[0] < 1 || cp->rank != 1 || cp->stride[0] != 1 ||
            cu->stride[0] != 1 || cp->shape[0] != cu->shape[0] || part > (1ll << 30) / o->shape[0] ||
            M < (part + 1) * o->shape[0])
            return R4D_NO(RAD_E_SHAPE);
        row0 = part * o->shape[0];
        M = o->shape[0];
    } else if (cp) {
        return R4D_NO(RAD_E_INVAL);
    }
    /* The keys: the cache [pages, 1, 4, n/4] (a page `stride[0]` apart) or flat [G, n]. */
    long long pages = 0, key_ld = 0;
    if (k->rank == 4) {
        if (k->shape[1] != 1 || k->shape[2] * k->shape[3] != n || k->stride[3] != 1 ||
            k->stride[2] != k->shape[3] || k->stride[0] < n)
            return R4D_NO(RAD_E_SHAPE);
        pages = k->shape[0]; key_ld = k->stride[0];
    } else if (k->rank == 2) {
        if (k->shape[1] != n || k->stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        pages = k->shape[0]; key_ld = k->stride[0];
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (pages < 1) return R4D_NO(RAD_E_SHAPE);
    if (bt->rank != 2 || o->rank != 2 || cu->rank != 1 || cu->shape[0] < 2)
        return R4D_NO(RAD_E_SHAPE);
    if (bt->stride[1] != 1 || o->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long nseq = cu->shape[0] - 1;
    if (bt->shape[0] < nseq || r4d_numel(nc) < nseq || o->shape[0] != M || o->shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    const long long cols = bt->shape[1] < o->shape[1] ? bt->shape[1] : o->shape[1];
    const long long out_cols = o->shape[1];
    if (out_cols > (1ll << 20)) return R4D_NO(RAD_E_SHAPE);
    const float wscale = (float)(std::pow((double)H, -0.5) * std::pow((double)n, -0.5));
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    const bool tiled = g_idx_score_form == 2 || (g_idx_score_form == 0 && M >= kTileRows);
    const unsigned short* qp = (const unsigned short*)r4d_p(q);
    const unsigned short* gp = (const unsigned short*)r4d_p(g);
    const int32_t* cup = (const int32_t*)r4d_p(cu);
    if (part >= 0) {
        if (qp) qp += row0 * q->stride[0];
        if (gp) gp += row0 * g->stride[0];
        const int rc = R4D_EMIT(stream, idx_rebase_kernel, dim3(1), dim3(64), 0, cup,
                                (int32_t*)r4d_wp(cp), (int)cu->shape[0], (int)row0);
        if (rc != RAD_OK) return rc;
        cup = (const int32_t*)r4d_p(cp);
    }
#define R4D_ISC_ARGS                                                                            \
    qp, gp, (const unsigned short*)r4d_p(k), (const int32_t*)r4d_p(bt), cup,                    \
        (const int32_t*)r4d_p(nc), (float*)r4d_wp(o), (int)nseq, (int)cols, (int)out_cols,      \
        (int)pages, (long)q->stride[0], (long)g->stride[0], (long)key_ld, (long)bt->stride[0],  \
        (long)o->stride[0], wscale
    if (tiled) {
        const dim3 grid((unsigned)((M + kTRows - 1) / kTRows),
                        (unsigned)((out_cols + kTKeys - 1) / kTKeys));
#define R4D_ISC(CH_)                                                                            \
    R4D_EMIT(stream, (idx_score_tile_kernel<CH_>), grid, dim3(kTWaves * kWave), 0, R4D_ISC_ARGS, \
             (int)M)
        return chain ? R4D_ISC(r4dchain) : R4D_ISC(R4dUnchained);
#undef R4D_ISC
    }
    const unsigned gy = (unsigned)((out_cols + kKeysWg - 1) / kKeysWg);
#define R4D_ISC(CH_)                                                                            \
    R4D_EMIT(stream, (idx_score_kernel<CH_>), dim3((unsigned)M, gy), dim3(kWaves * kWave), 0,   \
             R4D_ISC_ARGS)
    return chain ? R4D_ISC(r4dchain) : R4D_ISC(R4dUnchained);
#undef R4D_ISC
#undef R4D_ISC_ARGS
}
