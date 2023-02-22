/**
 * @file Functions.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace Optimization {


inline std::vector<double> random_initial_position(std::vector<double> min_bound, std::vector<double> max_bound) {
    std::random_device rd;
    std::mt19937       gen(rd());
    std::vector<double> x(min_bound.size());
    for (std::size_t i = 0; i < min_bound.size(); ++i) {
        std::uniform_real_distribution<> dis(min_bound[i], max_bound[i]);
        x[i] = dis(gen);
    }
    return x;
}

inline std::vector<double> random_initial_position(std::vector<double> min_bound, std::vector<double> max_bound, int seed) {
    std::mt19937       gen(seed);
    std::vector<double> x(min_bound.size());
    for (std::size_t i = 0; i < min_bound.size(); ++i) {
        std::uniform_real_distribution<> dis(min_bound[i], max_bound[i]);
        x[i] = dis(gen);
    }
    return x;
}

}  // namespace Optimization