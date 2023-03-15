/**
 * @file Statistics.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-03-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>


namespace utils {


template <typename T>
T mean(const std::vector<T>& list_values) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute mean of empty vector");
    }
    return std::accumulate(list_values.begin(), list_values.end(), 0.0) / list_values.size();
}

template <typename T>
T variance(const std::vector<T>& list_values) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute variance of empty vector");
    }
    T mean_value = mean(list_values);
    T sum        = 0.0;
    for (const auto& value : list_values) {
        sum += (value - mean_value) * (value - mean_value);
    }
    return sum / list_values.size();
}

template <typename T>
T standard_deviation(const std::vector<T>& list_values) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute standard deviation of empty vector");
    }
    return std::sqrt(variance(list_values));
}

template <typename T>
T median(const std::vector<T>& list_values) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute median of empty vector");
    }
    std::vector<T> list_values_sorted = list_values;
    std::sort(list_values_sorted.begin(), list_values_sorted.end());
    if (list_values_sorted.size() % 2 == 0) {
        return (list_values_sorted[list_values_sorted.size() / 2 - 1] + list_values_sorted[list_values_sorted.size() / 2]) / 2.0;
    } else {
        return list_values_sorted[list_values_sorted.size() / 2];
    }
}

template <typename T>
T percentile(const std::vector<T>& list_values, double percentile) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute percentile of empty vector");
    }
    if (percentile < 0.0 || percentile > 100.0) {
        throw std::runtime_error("Percentile must be between 0 and 100");
    }
    std::vector<T> list_values_sorted = list_values;
    std::sort(list_values_sorted.begin(), list_values_sorted.end());
    return list_values_sorted[static_cast<std::size_t>(std::floor(percentile / 100.0 * list_values_sorted.size()))];
}

template <typename T>
T min(const std::vector<T>& list_values) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute min of empty vector");
    }
    return *std::min_element(list_values.begin(), list_values.end());
}

template <typename T>
T max(const std::vector<T>& list_values) {
    if (list_values.empty()) {
        throw std::runtime_error("Cannot compute max of empty vector");
    }
    return *std::max_element(list_values.begin(), list_values.end());
}

}  // namespace utils


