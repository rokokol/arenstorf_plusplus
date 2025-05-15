#pragma once

#include "matplotlibcpp.h"
#include "satellite.h"
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
    std::vector<satellite<m>> models;
    constexpr static double M = 1 - m;
    std::vector<std::vector<double>> collisions;

public:
    rung(std::vector<satellite<m>> _models)
        : models(_models)
        , collisions(_models.size() - 1)
    {}

    bool make_steps()
    {
        bool state = false;
        for (auto &model : models) {
            state |= !model.completed;
            if (!model.completed) [[likely]] {
                model.make_step();
            }
        }

        return state;
    }

    void plot_satellites()
    {
        plt::figure(1);
        plt::clf();
        plt::xlim(-2, 2);
        plt::ylim(-2, 2);

        plt::plot(
            {0}, {0}, {{"marker", "o"}, {"markersize", std::to_string(m * 20)}, {"color", "red"}});
        plt::plot(
            {1}, {0}, {{"marker", "o"}, {"markersize", std::to_string(M * 20)}, {"color", "blue"}});

        for (auto &model : models) {
            std::vector<double> x_list, y_list;
            x_list.reserve(model.coordinate_list.size());
            y_list.reserve(model.coordinate_list.size());

            for (const auto &[key, p] : model.coordinate_list) {
                x_list.push_back(p.first);
                y_list.push_back(p.second);
            }

            plt::plot({x_list.back()}, {y_list.back()}, "xk");
            plt::plot(x_list, y_list, "--");
        }

        plt::show();
    }

    auto check_collisions(double timeshift = 0)
    {
        for (int i = 0; i < models.size() - 1; i++) {
            collisions[i] = std::vector<double>(models.size() - i - 1, 0.0);
        }

        for (int i = 0; i < models.size() - 1; i++) {
            auto &model = models[i];
            for (int j = i + 1; j < models.size(); j++) {
                auto &other = models[j];
                auto model_i = model.coordinate_list.begin();
                auto other_i = other.coordinate_list.begin();
                // Если разница во времени между следующим шагом больше чем между текущим
                while (model_i != model.coordinate_list.end()
                       && other_i != other.coordinate_list.end()) {
                    // перематываем до момента timeshift
                    while ((++model_i)->first < timeshift)
                        ;
                    while ((++other_i)->first < timeshift)
                        ;

                    if (std::abs(std::next(model_i)->first - other_i->first)
                        > std::abs(model_i->first - other_i->first)) {
                        ++model_i;
                    }

                    // Просто эвклидово расстояние :череп
                    auto dist_sq = (model_i->second.first - other_i->second.first)
                                       * (model_i->second.first - other_i->second.first)
                                   + (model_i->second.second - other_i->second.second)
                                         * (model_i->second.second - other_i->second.second);

                    if (dist_sq > 0.00001)
                        continue;

                    // 8/(2+x)^3 в нуле равна 1, притом неотрицательна и убывает -- отличная плотность
                    double delta_time = std::abs(model_i->first - other_i->first);
                    auto denominator = 2 + delta_time;
                    collisions[i][j - 1] += 8 * (1 - collisions[i][j - 1])
                                            / (denominator * denominator * denominator);
                    ++other_i;
                }
            }
        }

        return collisions;
    }
};
