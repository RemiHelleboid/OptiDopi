/**
 * @file GradientDescent.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-02-21
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

#include "device.hpp"

namespace Optimization {

class GradientDescent {
 private:
    std::vector<double> m_x;
    std::vector<double> dx;
    std::vector<double> m_x_best;
    double              m_cost_best;

    std::function<double(std::vector<double>)> m_cost_function;

    std::size_t m_max_iter;
    std::size_t m_iter;
    double      m_tol;
    double      m_alpha;

 public:
    GradientDescent(std::function<double(std::vector<double>)> cost_function,
                    std::vector<double>                        x0,
                    double                                     tol,
                    std::size_t                                max_iter,
                    double                                     alpha)
        : m_cost_function(cost_function),
          m_x(x0),
          m_tol(tol),
          m_max_iter(max_iter),
          m_alpha(alpha) {
        m_cost_best = std::numeric_limits<double>::max();
        m_iter      = 0;
        dx.resize(m_x.size());
    }

    void optimize();
};

}  // namespace Optimization
