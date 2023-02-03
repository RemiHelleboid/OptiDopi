#pragma once

#include <iostream>
#include <numeric>
#include <vector>

namespace Utils {

/**
 * @brief Compute the gradient of a vector y with respect to a vector x.
 * Using the central difference formula in the interior and either forward or backward difference at the boundaries.
*/
template <typename T>
std::vector<T> gradient(const std::vector<T>& x, const std::vector<T>& y) {
    if (x.size() != y.size()) {
        std::cerr << "Error: x and y must have the same size." << std::endl;
        throw std::invalid_argument("Error: x and y must have the same size.");
    }
    std::vector<T> gradient;
    gradient.resize(x.size());
    gradient[0] = (y[1] - y[0]) / (x[1] - x[0]);
    for (std::size_t index = 1; index < x.size() - 1; ++index) {
        gradient[index] = (y[index + 1] - y[index - 1]) / (x[index + 1] - x[index - 1]);
    }
    gradient[x.size() - 1] = (y[x.size() - 1] - y[x.size() - 2]) / (x[x.size() - 1] - x[x.size() - 2]);
    return gradient;
}

} // namespace Utils
