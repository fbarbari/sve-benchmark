#include <chrono>
#include <random>
#include <vector>
#include <benchmark/benchmark.h>
#include <matmul.hpp>

#if defined(__GNUG__)
#define DO_NOT_OPTIMIZE (optimize("O0"))
#elif defined(__clang__)
#define DO_NOT_OPTIMIZE (optnone)
#else
#error "unknown compiler"
#endif

template <typename T>
static void init_matmul(std::vector<T> a, std::vector<T> b, std::vector<T> c) {
    std::mt19937 rnd{std::random_device{}()};
    std::uniform_real_distribution<T> dist{static_cast<T>(-100), static_cast<T>(100)};

    std::generate(a.begin(), a.end(), [&dist, &rnd]() { return dist(rnd); });
    std::generate(b.begin(), b.end(), [&dist, &rnd]() { return dist(rnd); });
    std::fill(c.begin(), c.end(), 0.0);
}

template <typename T>
__attribute(DO_NOT_OPTIMIZE) static void BM_MatMul_NoOpt(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);
    std::vector<T> a(m * n);
    std::vector<T> b(n * k);
    std::vector<T> c(m * k);

    for (auto _ : state) {
        init_matmul(a, b, c);
        const auto start = std::chrono::high_resolution_clock::now();

        matmul::matmul(a.data(), b.data(), c.data(), m, n, k);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

template <typename T>
static void BM_MatMul_AutoVec(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);
    std::vector<T> a(m * n);
    std::vector<T> b(n * k);
    std::vector<T> c(m * k);

    for (auto _ : state) {
        init_matmul(a, b, c);
        const auto start = std::chrono::high_resolution_clock::now();

        matmul::matmul(a.data(), b.data(), c.data(), m, n, k);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

template <typename T>
static void BM_MatMul_Hwy(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);
    std::vector<T> a(m * n);
    std::vector<T> b(n * k);
    std::vector<T> c(m * k);

    for (auto _ : state) {
        init_matmul(a, b, c);
        const auto start = std::chrono::high_resolution_clock::now();

        matmul::matmul_hwy(a.data(), b.data(), c.data(), m, n, k);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();

        const auto end = std::chrono::high_resolution_clock::now();
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

BENCHMARK(BM_MatMul_NoOpt<float>)->Arg(1 << 9)->UseManualTime();
BENCHMARK(BM_MatMul_NoOpt<double>)->Arg(1 << 9)->UseManualTime();
BENCHMARK(BM_MatMul_AutoVec<float>)->Arg(1 << 9)->UseManualTime();
BENCHMARK(BM_MatMul_AutoVec<double>)->Arg(1 << 9)->UseManualTime();
BENCHMARK(BM_MatMul_Hwy<float>)->Arg(1 << 9)->UseManualTime();
BENCHMARK(BM_MatMul_Hwy<double>)->Arg(1 << 9)->UseManualTime();
