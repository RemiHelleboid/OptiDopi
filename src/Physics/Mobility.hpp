/**
 * @brief 
 * 
 */

#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <string>

namespace physic {

namespace model {
/**
 * @brief Compute the electron mobility using Arora model for Silicon.
 *
 * @param doping_concentration
 * @param Temperature
 * @return double
 */
inline double electron_mobility_arora(double doping_concentration, double Temperature) {
    constexpr double REF_TEMPERATURE      = 300.0;
    constexpr double A_min                = 88.0;
    constexpr double alpha_min            = -0.57;
    constexpr double A_dop                = 1252.0;
    constexpr double alpha_dop            = -2.33;
    constexpr double A_N                  = 1.25e17;
    constexpr double alpha_N              = 2.4;
    constexpr double A_a                  = 0.88;
    constexpr double alpha_a              = -0.146;
    const double     relative_temperature = Temperature / REF_TEMPERATURE;
    const double     mu_min               = A_min * pow(relative_temperature, alpha_min);
    const double     mu_d                 = A_dop * pow(relative_temperature, alpha_dop);
    const double     N_0                  = A_N * pow(relative_temperature, alpha_N);
    const double     A_star               = A_a * pow(relative_temperature, alpha_a);
    const double     doping_coefficient   = pow((fabs(doping_concentration) / N_0), A_star);
    const double     mu_dop               = mu_d / (1 + doping_coefficient);
    const double     electron_mobility    = mu_min + mu_dop;
    return electron_mobility;
}

/**
 * @brief Compute the hole mobility using Arora model for Silicon.
 *
 * Length are in cm, volume in cm^-3.
 *
 * @param Temperature
 * @param doping_concentration
 * @return double
 */
inline double hole_mobility_arora(double doping_concentration, double Temperature) {
    constexpr double REF_TEMPERATURE      = 300.0;
    constexpr double A_min                = 54.3;
    constexpr double alpha_min            = -0.57;
    constexpr double A_dop                = 407.0;
    constexpr double alpha_dop            = -2.23;
    constexpr double A_N                  = 2.35e17;
    constexpr double alpha_N              = 2.4;
    constexpr double A_a                  = 0.88;
    constexpr double alpha_a              = -0.146;
    const double     relative_temperature = Temperature / REF_TEMPERATURE;
    const double     mu_min               = A_min * pow(relative_temperature, alpha_min);
    const double     mu_d                 = A_dop * pow(relative_temperature, alpha_dop);
    const double     N_0                  = A_N * pow(relative_temperature, alpha_N);
    const double     A_star               = A_a * pow(relative_temperature, alpha_a);
    const double     doping_coefficient   = pow((fabs(doping_concentration) / N_0), A_star);
    const double     mu_dop               = mu_d / (1 + doping_coefficient);
    const double     hole_mobility        = mu_min + mu_dop;
    return hole_mobility;
}

inline double electron_saturation_velocity(double temperature) {
    constexpr double REF_TEMPERATURE              = 300.0;
    constexpr double v_sat_min                    = 1.07e7;
    constexpr double v_sat_exponent               = 0.87;
    const double     electron_saturation_velocity = v_sat_min * pow((temperature / REF_TEMPERATURE), v_sat_exponent);
    return electron_saturation_velocity;
}

inline double hole_saturation_velocity(double temperature) {
    constexpr double REF_TEMPERATURE          = 300.0;
    constexpr double v_sat_min                = 8.37e6;
    constexpr double v_sat_exponent           = 0.52;
    const double     hole_saturation_velocity = v_sat_min * pow((temperature / REF_TEMPERATURE), v_sat_exponent);
    return hole_saturation_velocity;
}

/**
 * @brief Mobility for electron using Canali model for high fields for Silicon.
 *
 * @param doping_concentration
 * @param ElectricField
 * @param Temperature
 * @return double
 */
inline double electron_mobility_arora_canali(double doping_concentration, double ElectricField, double Temperature) {
    constexpr double REF_TEMPERATURE    = 300.0;
    constexpr double beta_0             = 1.109;
    constexpr double beta_exp           = 0.66;
    constexpr double alpha              = 0.0;
    const double     beta               = beta_0 * pow((Temperature / REF_TEMPERATURE), beta_exp);
    const double     mobility_low_field = electron_mobility_arora(doping_concentration, Temperature);
    const double     v_sat              = electron_saturation_velocity(Temperature);
    const double     nominator          = (alpha + 1) * mobility_low_field;
    const double     denominator = alpha + pow(1 + pow((((alpha + 1) * mobility_low_field * ElectricField) / v_sat), beta), (1.0 / beta));
    const double     electron_mobility = nominator / denominator;
    return electron_mobility;
}

/**
 * @brief Mobility for hole using Canali model for high fields for Silicon.
 *
 * @param doping_concentration
 * @param ElectricField
 * @param Temperature
 * @return double
 */
inline double hole_mobility_arora_canali(double doping_concentration, double ElectricField, double Temperature) {
    constexpr double REF_TEMPERATURE    = 300.0;
    constexpr double beta_0             = 1.213;
    constexpr double beta_exp           = 0.17;
    constexpr double alpha              = 0.0;
    const double     beta               = beta_0 * pow((Temperature / REF_TEMPERATURE), beta_exp);
    const double     mobility_low_field = hole_mobility_arora(doping_concentration, Temperature);
    const double     v_sat              = hole_saturation_velocity(Temperature);
    const double     nominator          = (alpha + 1) * mobility_low_field;
    const double     denominator   = alpha + pow(1 + pow((((alpha + 1) * mobility_low_field * ElectricField) / v_sat), beta), (1.0 / beta));
    const double     hole_mobility = nominator / denominator;
    return hole_mobility;
}

} // namespace model

} // namespace physics

