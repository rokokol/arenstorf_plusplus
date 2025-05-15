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
    std::vector<satellite<>> models{
        satellite<>(0.994, 0.0, 0.0, -2.031732629557337),
        satellite<>(0.994, 0.0, 0.0, -2.00158510637908252240537862224),
        satellite<>(1.2, 0.0, 0.0, -1.5),
        // satellite<>(0.8, 0.0, 0.0, 2.5),
        // satellite<>(0.994, 0.1, 0.2, -2),
    };
    rung test(models); // 0.994, 0.0, 0.0, -2.031732629557337, 0.05, 0.1
    while (test.make_steps());
    // printf("colission proba: %.16f\n", test.check_collisions(0.1)[0][0]);
    test.plot_satellites();
}

int main()
{
    measure_execution_time(merson_solve);
    // plt::plot(state{3, 4, 1}, state{6, 3, 1});
    // plt::show();
    return 0;
}
