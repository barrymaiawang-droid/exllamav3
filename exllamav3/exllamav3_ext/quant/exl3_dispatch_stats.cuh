#pragma once
/*
Dispatch counters for the ROCm experiment build (not upstream).

Off unless EXL3_DISPATCH_STATS is set and not "0". The flag is read once and every increment sits behind
it, so a run with the variable unset issues the same launches and the same arithmetic and adds one
predictable branch per dispatch. Counters are host-side and count launches: under CUDA-graph capture the
capture is counted and later replays are not, so a graph-replaying run undercounts relative to tokens
(reported alongside the profiler trace, which sees the replayed kernels).

The set is exactly the paths this experiment must tell apart on gfx1151: dense decode GEMV (m = 1), the
multirow verifier (m = 2..8, the MTP batch), the cooperative prefill GEMM, routed MoE fused decode, and the
alternative each of those falls back to.
*/
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace exl3_dispatch_stats
{

enum Counter
{
    GEMM_RDNA_M1 = 0, GEMM_RDNA_M2, GEMM_RDNA_M3, GEMM_RDNA_M4,
    GEMM_RDNA_M5, GEMM_RDNA_M6, GEMM_RDNA_M7, GEMM_RDNA_M8,
    GEMM_RDNA_GEMV_EAGER,
    GEMM_RDNA_GEMV_GRAPH,
    GEMM_INT8_GEMV,
    GEMM_COOP,
    MGEMV_RDNA_MULTIROW,
    MGEMV_RDNA_PLAIN,
    MGEMV_COOP,
    MGEMV_SLICED_COOP,
    MOE_RDNA_DECODE,
    MOE_KERNEL_FALLBACK,
    N_COUNTERS
};

inline const char* const NAMES[N_COUNTERS] =
{
    "gemm_rdna_m1", "gemm_rdna_m2", "gemm_rdna_m3", "gemm_rdna_m4",
    "gemm_rdna_m5", "gemm_rdna_m6", "gemm_rdna_m7", "gemm_rdna_m8",
    "gemm_rdna_gemv_eager", "gemm_rdna_gemv_graph", "gemm_int8_gemv",
    "gemm_coop_prefill", "mgemv_rdna_multirow", "mgemv_rdna_plain",
    "mgemv_coop", "mgemv_sliced_coop", "moe_rdna_decode", "moe_kernel_fallback"
};

inline std::atomic<long> COUNTERS[N_COUNTERS] = {};

inline bool enabled()
{
    static const bool e = []()
    {
        const char* v = std::getenv("EXL3_DISPATCH_STATS");
        return v && std::strcmp(v, "0") != 0;
    }();
    return e;
}

inline void bump(int counter, long n = 1)
{
    if (enabled()) COUNTERS[counter].fetch_add(n, std::memory_order_relaxed);
}

inline void bump_m(int m, int base = GEMM_RDNA_M1)
{
    if (m >= 1 && m <= 8) bump(base + (m - 1));
}

inline std::vector<std::pair<std::string, long>> snapshot()
{
    std::vector<std::pair<std::string, long>> out;
    out.reserve(N_COUNTERS);
    for (int i = 0; i < N_COUNTERS; ++i)
        out.emplace_back(NAMES[i], COUNTERS[i].load(std::memory_order_relaxed));
    return out;
}

inline void reset()
{
    for (int i = 0; i < N_COUNTERS; ++i) COUNTERS[i].store(0, std::memory_order_relaxed);
}

}   // namespace exl3_dispatch_stats
