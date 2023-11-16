#pragma once

#include <hwy/highway.h>

namespace matmul {

template <typename T>
void matmul(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c, size_t m, size_t n,
            size_t k) {
    for (size_t i{0u}; i < m; i++) {
        for (size_t l{0u}; l < k; l++) {
            const T ail = a[i * k + l];
            for (size_t j{0u}; j < n; j++) {
                c[i * n + j] += ail * b[l * n + j];
            }
        }
    }
}

template <typename T>
HWY_ATTR void matmul_hwy(const T* HWY_RESTRICT a, const T* HWY_RESTRICT b, T* HWY_RESTRICT c,
                         size_t m, size_t n, size_t k) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    using M = hn::Mask<decltype(d)>;
    const size_t lanes = hn::Lanes(d);

    for (size_t i{0u}; i < m; i++) {
        for (size_t l{0u}; l < k; l++) {
            const V ail = hn::Set(d, a[i * k + l]);

            size_t j{0u};
            for (; j + (lanes - 1) < n; j += lanes) {
                const V blj = hn::LoadU(d, b + (l * n + j));
                V cij = hn::LoadU(d, c + (i * n + j));
                cij = hn::MulAdd(ail, blj, cij);
                hn::StoreU(cij, d, c + (i * n + j));
            }
            if (j < n) {
                // remainder loop
                const M mask = hn::FirstN(d, n - j);
                const V blj = hn::MaskedLoad(mask, d, b + (l * n + j));
                V cij = hn::MaskedLoad(mask, d, c + (i * n + j));
                cij = hn::MulAdd(ail, blj, cij);
                hn::BlendedStore(cij, mask, d, c + (i * n + j));
            }
        }
    }
}

}  // namespace matmul
