#pragma once

// Force inline
#if defined(__CUDACC__)
#define TTNTE_INLINE __forceinline__
#else
#define TTNTE_INLINE __attribute__((always_inline)) inline
#endif
