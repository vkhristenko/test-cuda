#include <iostream>
#include <functional>
#include <cstdio>

#include <nvfunctional>

template<typename T, typename Functor>
struct TestF {
    T* data;
    Functor idxF;
    TestF(T* data, Functor f) : data{data}, idxF{f} {}

    T& operator()(int i, int j) {
        auto const idx = idxF(i, j);
        return data[i];
    }
};

struct IndexFunctor {
    __forceinline__
    static constexpr int index(int i, int j) { return 10*i +j; }
};

template<typename T, typename Functor>
struct TestFCUDA : public Functor {
    T* data;
    __forceinline__ __device__
    TestFCUDA(T* data) : data{data} {}

    __forceinline__ __device__
    T& operator()(int i, int j) {
        auto const idx = Functor::index(i, j);
        return data[i];
    }
};

__global__
void test_kernel() {
//    printf("tid = %d\n", threadIdx.x);
//    auto l = [](int i) -> int { return i;};
//    printf(" i = %d\n", l(10));
    int data[1];

    TestFCUDA<int, IndexFunctor> map{data};
    map(0, 0) = 10;
    printf("value = %d\n", map(0, 0));
}

int main() {
    printf("hello\n");

    int data[100];
    TestF<int, std::function<int(int, int)>> tf{data, [](int i, int j) -> int { return i+j; }};
    tf(0, 0) = 5;

    test_kernel<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
