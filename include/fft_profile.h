#ifndef FFT_PROFILE_H
#define FFT_PROFILE_H

/*
 * Profiling instrumentation, compiled in only when FFT_NVTX is defined
 * (see the `bench_prof` target in the Makefile). Without it every macro
 * below expands to nothing, so the shipping build carries no overhead.
 *
 *   FFT_RANGE_PUSH/POP   named NVTX range; shows up as a row in the
 *                        Nsight Systems timeline and as a group in
 *                        `nsys stats --report nvtx_pushpop_sum`.
 *   FFT_MARK             instantaneous NVTX marker.
 *   FFT_CAPTURE_BEGIN/END  cudaProfilerStart/Stop. Nsight Compute is run
 *                        with `--profile-from-start off`, so only kernels
 *                        launched between these two calls are replayed and
 *                        measured: the steady-state timed loop, not the
 *                        warmup or the correctness checks around it.
 *
 * NVTX comes from the header-only nvtx3 distribution that ships with the
 * CUDA Toolkit (11.0+), so there is no library to link.
 */

#ifdef FFT_NVTX
#include <nvtx3/nvToolsExt.h>
#define FFT_RANGE_PUSH(name) nvtxRangePushA(name)
#define FFT_RANGE_POP()      nvtxRangePop()
#define FFT_MARK(name)       nvtxMarkA(name)
#else
#define FFT_RANGE_PUSH(name) ((void)0)
#define FFT_RANGE_POP()      ((void)0)
#define FFT_MARK(name)       ((void)0)
#endif

#if defined(FFT_NVTX) && defined(__CUDACC__)
#include <cuda_profiler_api.h>
#define FFT_CAPTURE_BEGIN() ((void)cudaProfilerStart())
#define FFT_CAPTURE_END()   ((void)cudaProfilerStop())
#else
#define FFT_CAPTURE_BEGIN() ((void)0)
#define FFT_CAPTURE_END()   ((void)0)
#endif

#endif
