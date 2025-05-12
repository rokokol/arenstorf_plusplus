#pragma once

#include "matplotlibcpp.h"
#include <format>
#include <functional>
#include <immintrin.h>
#include <vector>
namespace plt = matplotlibcpp;

using state = std::vector<double>;
using ode_function = std::function<__m256d(double, const __m256d &)>;

template<double m = 0.012277471>
class rung
{
private:
    inline static double t = 0; // Почему не сбрасывается при создании нового объекта?
    double prev_t = t;
    double error{};
    double global_error{};
    double global_error_x{};
    double global_error_y{};
    double hmin = 1. / 100'000;
    double hmax = 1. / 1'000;
    double h = 0.0001;
    const double M = 1 - m;
    double emin;
    double emax;
    __m256d current_point;

public:
    state x_list;
    state y_list;
    state error_list;
    state time_list;

    rung(
        const double x0,
        const double y0,
        const double dx0,
        const double dy0,
        const double emin,
        const double emax)
        : emin(emin)
        , emax(emax)
    {
        current_point = _mm256_set_pd(dy0, dx0, y0, x0);
    }

    rung()
        : rung(0.994, 0.0, 0.0, -2.031732629557337, 0.05, 0.1)
    {}

    void make_step()
    {
        __m256d res = step();
        while (error > emax && h > hmin) {
            h /= 10.;
            res = step();
            std::cout << "t decrease: " << t << " h: " << h << std::endl;
        }

        t += h;

        if (error < emin && h < hmax) {
            h *= 10;
            std::cout << "t raise: " << t << " h: " << h << std::endl;
        }

        current_point = res;
        global_error += error * h;

        if (std::abs(prev_t - t) > 1. / 20) {
            alignas(32) double res_arr[4];
            _mm256_store_pd(res_arr, res);

            x_list.push_back(res_arr[0]);
            y_list.push_back(res_arr[1]);
            error_list.push_back(error);
            time_list.push_back(t);
            plot_satellite();
            prev_t = t;
        }
    }

    void plot_satellite()
    {
        plt::figure(1);
        plt::clf();
        plt::xlim(-2, 2);
        plt::ylim(-2, 2);

        plt::plot(
            {0}, {0}, {{"marker", "o"}, {"markersize", std::to_string(m * 20)}, {"color", "red"}});
        plt::plot(
            {1}, {0}, {{"marker", "o"}, {"markersize", std::to_string(M * 20)}, {"color", "blue"}});
        plt::plot({x_list.back()}, {y_list.back()}, "xk");
        plt::plot(x_list, y_list, "--");

        plt::plot(
            std::vector<double>{0.0},
            std::vector<double>{0.0},
            {{"linestyle", "None"},
             {"marker", "None"},
             {"label", std::format("error: {:.6f}", error)}});

        plt::plot(
            std::vector<double>{0.0},
            std::vector<double>{0.0},
            {{"linestyle", "None"},
             {"marker", "None"},
             {"label", std::format("global error: {:.10f}", global_error)}});

        plt::legend();
        plt::pause(h);
    }

private:
    // y = { x, y, v1, v2 }
    [[nodiscard]] __m256d arenstorf(double t, const __m256d &input) const
    {
        alignas(32) double input_arr[4];
        _mm256_store_pd(input_arr, input);
        const double x = input_arr[0];
        const double y = input_arr[1];
        const double dx = input_arr[2]; // v1
        const double dy = input_arr[3]; // v2

        alignas(32) double res[4];
        res[0] = dx;
        res[1] = dy;

        double r1_temp = (x + m) * (x + m) + y * y;
        double r1 = std::sqrt(r1_temp) * r1_temp;

        double r2_temp = (x - M) * (x - M) + y * y;
        double r2 = std::sqrt(r2_temp) * r2_temp;

        res[2] = x + 2 * dy - M * (x + m) / r1 - m * (x - M) / r2;
        res[3] = y - 2 * dx - M * y / r1 - m * y / r2;

        return _mm256_load_pd(res); // diff: x y dx dy
    }

    __m256d step()
    {
        // Вычисление k1 = f(t, y0)
        const __m256d k1 = arenstorf(t, current_point);

        // Вычисление k2 = f(t + h/3, y0 + (h/3)*k1)
        __m256d y_temp = _mm256_add_pd(current_point, _mm256_mul_pd(_mm256_set1_pd(h / 3.0), k1));
        const __m256d k2 = arenstorf(t + h / 3.0, y_temp);

        // Вычисление k3 = f(t + h/3, y0 + (h/6)*(k1 + k2))
        y_temp = _mm256_add_pd(
            current_point, _mm256_mul_pd(_mm256_set1_pd(h / 6.0), _mm256_add_pd(k1, k2)));
        const __m256d k3 = arenstorf(t + h / 3.0, y_temp);

        // Вычисление k4 = f(t + h/2, y0 + (h/8)*(k1 + 3*k3))
        y_temp = _mm256_add_pd(
            current_point,
            _mm256_mul_pd(
                _mm256_set1_pd(h / 8.0), _mm256_add_pd(k1, _mm256_mul_pd(k3, _mm256_set1_pd(3.0)))));
        const __m256d k4 = arenstorf(t + h / 2.0, y_temp);

        // Вычисление k5 = f(t + h, y0 + (h/2)*(k1 - 3*k3 + 4*k4))
        y_temp = _mm256_add_pd(
            current_point,
            _mm256_mul_pd(
                _mm256_set1_pd(h / 2.0),
                _mm256_add_pd(
                    _mm256_add_pd(k1, _mm256_mul_pd(k3, _mm256_set1_pd(-3.0))),
                    _mm256_mul_pd(k4, _mm256_set1_pd(4.0)))));
        const __m256d k5 = arenstorf(t + h, y_temp);

        // Итоговое приближение: y_next = y0 + (h/6)*(k1 + 4*k4 + k5)
        __m256d y_next = _mm256_add_pd(
            current_point,
            _mm256_mul_pd(
                _mm256_set1_pd(h / 6.0),
                _mm256_add_pd(k1, _mm256_add_pd(k5, _mm256_mul_pd(_mm256_set1_pd(4.0), k4)))));

        // Оценка локальной ошибки: для каждой компоненты вычисляем
        // e_i = |(-k1[i] + 4*k3[i] + k4[i] - k5[i]) / 30|, затем берём максимум
        __m256d errors = _mm256_mul_pd(
            _mm256_set1_pd(1 / 30.0),
            _mm256_add_pd(
                _mm256_sub_pd(_mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(4.0), k3), k1), k5), k4));
        const __m256d mask = _mm256_castsi256_pd(
            _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF) // сбрасывание знака у double
        );

        errors = _mm256_and_pd(errors, mask);
        double error_list[4];
        _mm256_store_pd(error_list, errors);

        __m128d low = _mm256_castpd256_pd128(errors);
        __m128d high = _mm256_extractf128_pd(errors, 1);
        __m128d max1 = _mm_max_pd(low, high);

        // Вычисляем максимум из max1
        __m128d shuffled = _mm_shuffle_pd(max1, max1, 0b01);
        __m128d final = _mm_max_sd(max1, shuffled);
        error = _mm_cvtsd_f64(final); // Перезаписываем максимальную ошибку

        return y_next;
    }
};
