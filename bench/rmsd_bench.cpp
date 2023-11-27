#include <chrono>
#include <random>
#include <vector>
#include <benchmark/benchmark.h>
#include <rmsd.hpp>

#if defined(__GNUG__)
#define DO_NOT_OPTIMIZE (optimize("O0"))
#elif defined(__clang__)
#define DO_NOT_OPTIMIZE (optnone)
#else
#error "unknown compiler"
#endif

template <typename T>
static void init_rmsd(std::vector<T> x, std::vector<T> y, std::vector<T> z) {
    std::mt19937 rnd{std::random_device{}()};
    std::uniform_real_distribution<T> dist{static_cast<T>(-100), static_cast<T>(100)};

    std::generate(x.begin(), x.end(), [&dist, &rnd]() { return dist(rnd); });
    std::generate(y.begin(), y.end(), [&dist, &rnd]() { return dist(rnd); });
    std::fill(z.begin(), z.end(), static_cast<T>(0));
}

template <typename T>
__attribute(DO_NOT_OPTIMIZE) static void BM_RMSD_NoOpt(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<T> x(n);
    std::vector<T> y(n);
    std::vector<T> z(n);

    for (auto _ : state) {
        init_rmsd(x, y, z);
        const auto start = std::chrono::high_resolution_clock::now();

        auto result = rmsd::rmsd(x.data(), y.data(), z.data(), n);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

template <typename T>
static void BM_RMSD_AutoVec(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<T> x(n);
    std::vector<T> y(n);
    std::vector<T> z(n);

    for (auto _ : state) {
        init_rmsd(x, y, z);
        const auto start = std::chrono::high_resolution_clock::now();

        auto result = rmsd::rmsd(x.data(), y.data(), z.data(), n);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

template <typename T>
static void BM_RMSD_Hwy(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<T> x(n);
    std::vector<T> y(n);
    std::vector<T> z(n);

    for (auto _ : state) {
        init_rmsd(x, y, z);
        const auto start = std::chrono::high_resolution_clock::now();

        auto result = rmsd::rmsd_hwy(x.data(), y.data(), z.data(), n);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

BENCHMARK(BM_RMSD_NoOpt<float>)->Arg(1 << 20)->UseManualTime();
BENCHMARK(BM_RMSD_NoOpt<double>)->Arg(1 << 20)->UseManualTime();
BENCHMARK(BM_RMSD_AutoVec<float>)->Arg(1 << 20)->UseManualTime();
BENCHMARK(BM_RMSD_AutoVec<double>)->Arg(1 << 20)->UseManualTime();
BENCHMARK(BM_RMSD_Hwy<float>)->Arg(1 << 20)->UseManualTime();
BENCHMARK(BM_RMSD_Hwy<double>)->Arg(1 << 20)->UseManualTime();
