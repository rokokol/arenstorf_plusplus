#include "matplotlibcpp.h"
#include "rung.h"
#include <chrono>
#include <deque>
#include <functional>
#include <immintrin.h>
namespace plt = matplotlibcpp;

template<typename F, typename... Args>
void measure_execution_time(F &&func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void merson_solve()
{
    rung test{};
    while(true) {
        test.make_step();
    }
}

int main()
{
    measure_execution_time(merson_solve);
    // plt::plot(state{3, 4, 1}, state{6, 3, 1});
    // plt::show();
    return 0;
}
