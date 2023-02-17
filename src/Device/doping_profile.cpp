/**
 * @file doping_profile.cpp
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "doping_profile.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "fill_vector.hpp"
#include "smoother.hpp"

std::pair<double, double> exponential_link_parameters(double x0, double x1, double y0, double y1) {
    double alpha = log(y1 / y0) / (x1 - x0);
    double beta  = y0 / exp(alpha * x0);
    return std::make_pair(alpha, beta);
}

double exponential_link(double x, double alpha, double beta) { return beta * exp(alpha * x); }

doping_profile::doping_profile(double x_min, double x_max, std::size_t number_points)
    : m_x_line(utils::linspace(x_min, x_max, number_points)),
      m_acceptor_concentration(number_points, 0.0),
      m_donor_concentration(number_points, 0.0) {
    re_compute_total_doping();
}

void doping_profile::smooth_doping_profile(int window_size) {
    m_donor_concentration    = Utils::convol_square(m_donor_concentration, window_size);
    m_acceptor_concentration = Utils::convol_square(m_acceptor_concentration, window_size);
    re_compute_total_doping();
}

void doping_profile::re_compute_total_doping() {
    if (m_acceptor_concentration.size() != m_donor_concentration.size()) {
        throw std::logic_error("Error: Acceptor and donor profile have different numbers of values. Cannot compute the total doping.");
    }
    m_doping_concentration.resize(m_acceptor_concentration.size());
    for (std::size_t index_value = 0; index_value < m_acceptor_concentration.size(); index_value++) {
        m_doping_concentration[index_value] = -m_donor_concentration[index_value] + m_acceptor_concentration[index_value];
    }
}

void doping_profile::set_up_pin_diode(double      x_min,
                                      double      x_max,
                                      std::size_t number_points,
                                      double      length_donor,
                                      double      length_intrinsic,
                                      double      donor_level,
                                      double      acceptor_level,
                                      double      intrisic_level) {
    m_x_line = utils::linspace(x_min, x_max, number_points);
    m_donor_concentration.clear();
    m_acceptor_concentration.clear();
    m_donor_concentration.resize(number_points);
    m_acceptor_concentration.resize(number_points);

    for (std::size_t index_x = 0; index_x < number_points; ++index_x) {
        double x_position = m_x_line[index_x];
        if (x_position <= length_donor) {
            m_donor_concentration[index_x]    = donor_level;
            m_acceptor_concentration[index_x] = 0.0;
        } else if (x_position <= length_donor + length_intrinsic) {
            m_donor_concentration[index_x]    = intrisic_level;
            m_acceptor_concentration[index_x] = 0.0;
        } else {
            m_donor_concentration[index_x]    = 0.0;
            m_acceptor_concentration[index_x] = acceptor_level;
        }
    }
    re_compute_total_doping();
}

void doping_profile::set_up_advanced_pin(double              xlength,
                                         std::size_t         number_points,
                                         double              length_donor,
                                         double              length_intrinsic,
                                         double              donor_level,
                                         double              intrisic_level,
                                         std::vector<double> list_x_acceptor,
                                         std::vector<double> list_acceptor_level) {
    if (list_x_acceptor.size() != list_acceptor_level.size()) {
        throw std::logic_error("Error: Acceptor and donor profile have different numbers of values. Cannot compute the total doping.");
    }
    if (list_x_acceptor.back() > xlength) {
        throw std::logic_error("Error: Acceptor and donor profile have different numbers of values. Cannot compute the total doping.");
    }
    std::cout << "Setting up advanced pin diode" << std::endl;

    m_x_line = utils::linspace(0.0, xlength, number_points);
    m_donor_concentration.clear();
    m_acceptor_concentration.clear();
    m_donor_concentration.resize(number_points);
    m_acceptor_concentration.resize(number_points);

    for (std::size_t index_x = 0; index_x < number_points; ++index_x) {
        double x_position = m_x_line[index_x];
        if (x_position <= length_donor) {
            m_donor_concentration[index_x]    = donor_level;
            m_acceptor_concentration[index_x] = 0.0;
        } else if (x_position <= length_donor + length_intrinsic) {
            m_donor_concentration[index_x]    = intrisic_level;
            m_acceptor_concentration[index_x] = 0.0;
        } else {
            break;
        }
    }
    int nb_acceptor = list_x_acceptor.size();

    for (std::size_t idx_acc = 0; idx_acc < nb_acceptor; ++idx_acc) {
        double x_init     = list_x_acceptor[idx_acc];
        double x_end      = list_x_acceptor[idx_acc + 1];
        double y_init     = list_acceptor_level[idx_acc];
        double y_end      = list_acceptor_level[idx_acc + 1];
        auto   alpha_beta = exponential_link_parameters(x_init, x_end, y_init, y_end);
        double alpha      = alpha_beta.first;
        double beta       = alpha_beta.second;
        for (std::size_t index_x = 0; index_x < number_points; ++index_x) {
            double x_position = m_x_line[index_x];
            if (x_position >= x_init && x_position <= x_end) {
                m_acceptor_concentration[index_x] = exponential_link(x_position, alpha, beta);
            }
        }
    }
    re_compute_total_doping();
}

void doping_profile::export_doping_profile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::logic_error("Error: Cannot open file " + filename + " for writing.");
    }
    file << "x_position" << ',' << "donor_concentration" << ',' << "acceptor_concentration" << ',' << "total_doping" << '\n';
    for (std::size_t index_x = 0; index_x < m_x_line.size(); ++index_x) {
        file << m_x_line[index_x] << ',' << m_donor_concentration[index_x] << ',' << m_acceptor_concentration[index_x] << ','
             << m_doping_concentration[index_x] << '\n';
    }
    file.close();
}
