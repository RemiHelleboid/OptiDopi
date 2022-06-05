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

#include "fill_vector.hpp"

doping_profile::doping_profile(double x_min, double x_max, std::size_t number_points)
    : m_x_line(utils::linspace(x_min, x_max, number_points)),
      m_acceptor_concentration(number_points, 0.0),
      m_donor_concentration(number_points, 0.0) {
    re_compute_total_doping();
}

void doping_profile::re_compute_total_doping() {
    if (m_acceptor_concentration.size() != m_donor_concentration.size()) {
        throw std::logic_error("Error: Acceptor and donor profile have different numbers of values. Cannot compute the total doping.");
    }
    for (std::size_t index_value = 0; index_value < m_acceptor_concentration.size(); index_value++) {
        m_doping_concentration[index_value] = m_donor_concentration[index_value] - m_acceptor_concentration[index_value];
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
        } else if (x_position <= length_intrinsic) {
            m_donor_concentration[index_x]    = intrisic_level;
            m_acceptor_concentration[index_x] = intrisic_level;
        } else {
            m_donor_concentration[index_x]    = 0.0;
            m_acceptor_concentration[index_x] = acceptor_level;
        }
    }
    re_compute_total_doping();
}
