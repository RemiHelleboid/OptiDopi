/**
 * @file GradientDescent.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>

#include <fmt/core.h>
#include <fmt/ranges.h>


#include "GradientDescent.hpp"

namespace Optimization {

/**
 * @brief Optimize the cost function using the gradient descent method.
 * No log is saved.
 * 
 */
void GradientDescent::optimize() {
    double cost = m_cost_function(m_x);
    while (m_iter < m_max_iter) {
        std::cout << "Iteration: " << m_iter << " Cost: " << cost << std::endl;
        // Compute the gradient
        for (std::size_t i = 0; i < m_x.size(); ++i) {
            std::vector<double> x_plus(m_x);
            x_plus[i] += m_alpha;
            std::vector<double> x_minus(m_x);
            x_minus[i] -= m_alpha;
            dx[i] = (m_cost_function(x_plus) - m_cost_function(x_minus)) / (2 * m_alpha);
        }
        // Print the gradient
        fmt::print("Gradient: [{}]\n", fmt::join(dx, ", "));
        // Update the position
        for (std::size_t i = 0; i < m_x.size(); ++i) {
            m_x[i] -= dx[i];
        }
        cost = m_cost_function(m_x);
        ++m_iter;
    }

    // Save the best position
    if (cost < m_cost_best) {
        m_cost_best = cost;
        m_x_best    = m_x;
    } 

}
    




}  // namespace Optimization
