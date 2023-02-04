/**
 * @file fill_vector.hpp
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <algorithm>
#include <iostream>
#include <vector>

namespace utils {

template <typename T>
std::vector<T> linspace(T x_min, T x_max, std::size_t number_points) {
    std::vector<T> list_x;
    list_x.resize(number_points);
    if (number_points == 0) {
        return list_x;
    }
    if (number_points == 1) {
        list_x[0] = x_min;
        return list_x;
    }
    double dx = (x_max - x_min) / (number_points - 1);
    for (std::size_t index_value = 0; index_value < number_points; ++index_value) {
        list_x[index_value] = x_min + dx * index_value;
    }
    return list_x;
}

template <typename T>
std::vector<T> geomspace(T x_min, T x_max, std::size_t number_points) {
    std::vector<T> list_x;
    list_x.resize(number_points);
    if (number_points == 0) {
        return list_x;
    }
    if (number_points == 1) {
        list_x[0] = x_min;
        return list_x;
    }
    double dx = std::pow(x_max / x_min, 1.0 / (number_points - 1));
    for (std::size_t index_value = 0; index_value < number_points; ++index_value) {
        list_x[index_value] = x_min * std::pow(dx, index_value);
    }
    return list_x;
}

}  // namespace utils