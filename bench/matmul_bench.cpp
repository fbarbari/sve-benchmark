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

static void init_matmul(std::vector<double> a, std::vector<double> b, std::vector<double> c) {
    std::mt19937 rnd{std::random_device{}()};
    std::uniform_real_distribution<double> dist{-100.0, 100.0};

    std::generate(a.begin(), a.end(), [&dist, &rnd]() { return dist(rnd); });
    std::generate(b.begin(), b.end(), [&dist, &rnd]() { return dist(rnd); });
    std::fill(c.begin(), c.end(), 0.0);
}

__attribute(DO_NOT_OPTIMIZE) static void BM_MatMul_NoOpt(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);
    std::vector<double> a(m * n);
    std::vector<double> b(n * k);
    std::vector<double> c(m * k);

    init_matmul(a, b, c);

    for (auto _ : state) {
        matmul::matmul(a.data(), b.data(), c.data(), m, n, k);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

static void BM_MatMul_AutoVec(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);
    std::vector<double> a(m * n);
    std::vector<double> b(n * k);
    std::vector<double> c(m * k);

    init_matmul(a, b, c);

    for (auto _ : state) {
        matmul::matmul(a.data(), b.data(), c.data(), m, n, k);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

static void BM_MatMul_Hwy(benchmark::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(0);
    const size_t k = state.range(0);
    std::vector<double> a(m * n);
    std::vector<double> b(n * k);
    std::vector<double> c(m * k);

    init_matmul(a, b, c);

    for (auto _ : state) {
        matmul::matmul_hwy(a.data(), b.data(), c.data(), m, n, k);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_MatMul_NoOpt)->Arg(1 << 9);
BENCHMARK(BM_MatMul_AutoVec)->Arg(1 << 9);
BENCHMARK(BM_MatMul_Hwy)->Arg(1 << 9);
