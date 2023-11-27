#pragma once

#include <hwy/highway.h>

namespace rmsd {

template <typename T>
T rmsd(const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ z, size_t length) {
    // find mean point
    T mx{0};
    T my{0};
    T mz{0};
    for (size_t i{0u}; i < length; i++) {
        mx += x[i];
        my += y[i];
        mz += z[i];
    }
    mx /= static_cast<T>(length);
    my /= static_cast<T>(length);
    mz /= static_cast<T>(length);

    // compute rmsd
    T sum{0};
    for (size_t i{0u}; i < length; i++) {
        const T dx = x[i] - mx;
        const T dy = y[i] - my;
        const T dz = z[i] - mz;
        sum += dx * dx + dy * dy + dz * dz;
    }

    return std::sqrt(sum / static_cast<T>(length));
}

template <typename T>
HWY_ATTR T rmsd_hwy(const T* HWY_RESTRICT x, const T* HWY_RESTRICT y, const T* HWY_RESTRICT z,
                    size_t length) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    using V = hn::Vec<decltype(d)>;
    const size_t lanes = hn::Lanes(d);

    // find mean point
    V mx = hn::Zero(d);
    V my = hn::Zero(d);
    V mz = hn::Zero(d);

    {
        size_t i{0u};
        for (; i + (lanes - 1) < length; i += lanes) {
            const V xi = hn::LoadU(d, x + i);
            const V yi = hn::LoadU(d, y + i);
            const V zi = hn::LoadU(d, z + i);
            mx = hn::Add(mx, xi);
            my = hn::Add(my, yi);
            mz = hn::Add(mz, zi);
        }
        if (i < length) {
            const V xi = hn::LoadN(d, x + i, length - i);
            const V yi = hn::LoadN(d, y + i, length - i);
            const V zi = hn::LoadN(d, z + i, length - i);
            mx = hn::Add(mx, xi);
            my = hn::Add(my, yi);
            mz = hn::Add(mz, zi);
        }
        mx = hn::Set(d, hn::GetLane(hn::SumOfLanes(d, mx)) / static_cast<T>(length));
        my = hn::Set(d, hn::GetLane(hn::SumOfLanes(d, my)) / static_cast<T>(length));
        mz = hn::Set(d, hn::GetLane(hn::SumOfLanes(d, mz)) / static_cast<T>(length));
    }

    // compute rmsd
    V sum = hn::Zero(d);
    {
        size_t i{0u};
        for (; i + (lanes - 1) < length; i += lanes) {
            const V xi = hn::LoadU(d, x + i);
            const V yi = hn::LoadU(d, y + i);
            const V zi = hn::LoadU(d, z + i);
            const V dx = hn::Sub(xi, mx);
            const V dy = hn::Sub(yi, my);
            const V dz = hn::Sub(zi, mz);
            sum = hn::MulAdd(dx, dx, hn::MulAdd(dy, dy, hn::MulAdd(dz, dz, sum)));
        }
        if (i < length) {
            const V xi = hn::LoadN(d, x + i, length - i);
            const V yi = hn::LoadN(d, y + i, length - i);
            const V zi = hn::LoadN(d, z + i, length - i);
            const V dx = hn::Sub(xi, mx);
            const V dy = hn::Sub(yi, my);
            const V dz = hn::Sub(zi, mz);
            sum = hn::MulAdd(dx, dx, hn::MulAdd(dy, dy, hn::MulAdd(dz, dz, sum)));
        }
    }

    return std::sqrt(hn::GetLane(hn::SumOfLanes(d, sum)) / static_cast<T>(length));
}

}  // namespace rmsd
