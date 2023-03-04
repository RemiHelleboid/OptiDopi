/**
 *
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace Utils {

template <typename T>
T interp1d(const std::vector<T> &x, const std::vector<T> &y, T xnew) {
    if (x.size() != y.size()) {
        throw std::runtime_error("x and y must have the same size");
    }
    if (x.size() < 2) {
        throw std::runtime_error("x and y must have at least 2 elements");
    }

    // Find the index of the first element in x that is greater than xnew
    auto it = std::upper_bound(x.begin(), x.end(), xnew);
    if (it == x.begin()) {
        return y[0];
    } else if (it == x.end()) {
        return y[x.size() - 1];
    } else {
        auto i = std::distance(x.begin(), it);
        return y[i - 1] + (y[i] - y[i - 1]) / (x[i] - x[i - 1]) * (xnew - x[i - 1]);
    }
}

/**
 * @brief C++ modern implementation of interp1d
 *
 * @tparam T
 * @param x
 * @param y
 * @param xnew
 * @return std::vector<T>
 */
template <typename T>
std::vector<T> interp1d(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &xnew) {
    std::vector<T> ynew;
    ynew.resize(xnew.size());
    for (std::size_t index = 0; index < xnew.size(); ++index) {
        ynew[index] = interp1d(x, y, xnew[index]);
    }
    return ynew;
}

/**
 * @brief Interpolate a sorted vector. Faster than interp1d.
 * Elements in x must be sorted in ascending order.
 * Elements in xnew must be sorted in ascending order.
 * Elements in xnew must be strictly within the range of x.
 * 
 * @tparam T 
 * @param x 
 * @param y 
 * @param xnew 
 * @return std::vector<T> 
 */
template <typename T>
std::vector<T> interp1dSorted(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &xnew) {
    if (x.size() != y.size()) {
        throw std::runtime_error("x and y must have the same size");
    }

    std::vector<T> y_new(xnew.size());
    std::size_t    indexCurrentX = 0;
    for (std::size_t index = 0; index < xnew.size(); ++index) {
        auto it = std::upper_bound(x.begin() + indexCurrentX, x.end(), xnew[index]);
        if (it == x.begin()) {
            y_new[index] = y[0];
        } else if (it == x.end()) {
            y_new[index] = y[x.size() - 1];
        } else {
            indexCurrentX = std::distance(x.begin(), it);
            y_new[index]  = y[indexCurrentX - 1] + (y[indexCurrentX] - y[indexCurrentX - 1]) / (x[indexCurrentX] - x[indexCurrentX - 1]) * (xnew[index] - x[indexCurrentX - 1]);
        }
    }
    return y_new;
}



}  // namespace Utils