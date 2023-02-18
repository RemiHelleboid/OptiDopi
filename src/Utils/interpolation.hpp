/**
 * 
*/

#pragma once


#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>


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

template <typename T>
std::vector<T> interp1d(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &xnew) {
    std::vector<T> ynew;
    ynew.resize(xnew.size());
    for (int index = 0; index < xnew.size(); ++index) {
        int i = 0;
        while (x[i] < xnew[index]) {
            ++i;
        }
        if (i == 0) {
            ynew[index] = y[0];
        } else if (i == x.size()) {
            ynew[index] = y[x.size() - 1];
        } else {
            ynew[index] = y[i - 1] + (y[i] - y[i - 1]) / (x[i] - x[i - 1]) * (xnew[index] - x[i - 1]);
        }
    }
    return ynew;
}


}  // namespace Utils