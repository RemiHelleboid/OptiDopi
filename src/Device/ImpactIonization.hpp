/*! \file ImpacIonizationModel.h
 *  \brief Header file of impact ionizations rates models implementation.
 */

#pragma once


#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Physics.hpp"

namespace mcintyre {

inline double compute_gamma(const double Temperature) {
   const double h_barre_omega = 0.063;
    double Gamma = pow(tanh(h_barre_omega / (2 * boltzmann_constant_eV * 300)) / tanh(h_barre_omega / (2 * boltzmann_constant_eV * Temperature)), 0.59);
    return Gamma;
}

inline double compute_band_gap(const double Temperature) {
    double E_g = 1.166 - ((4.73e-4 * pow(Temperature, 2.0)) / (Temperature + 636.0));
    return E_g;
}

/**
 * @brief Electron impact ionization rate.
 * 
 * @param F_ava (V/micron)
 * @param Gamma (dimensionless)
 * @param E_g (eV)
 * @return double (1/micron)
 */
inline double alpha_DeMan(double F_ava, const double Gamma, const double E_g) {
    constexpr double lambda_e          = 62e-8;
    constexpr double elementary_charge = 1.0;
    constexpr double E_threshold       = 0.0;
    constexpr double E_0               = 4.0e1;

    if (F_ava <= E_threshold) {
        return 0.0;
    } else if (F_ava <= E_0) {
        double a_e                 = 7.03e1;
        double beta                = 0.678925;
        double b_e                 = (beta * E_g) / (lambda_e * elementary_charge);
        double imapct_ionization_e = Gamma * a_e * exp(-Gamma * b_e / F_ava);
        return imapct_ionization_e;
    } else {
        double a_e                 = 7.03e1;
        double beta                = 0.678925;
        double b_e                 = (beta * E_g) / (lambda_e * elementary_charge);
        double imapct_ionization_e = Gamma * a_e * exp(-Gamma * b_e / F_ava);

        return imapct_ionization_e;
    }
}

inline double beta_DeMan(double F_ava, const double Gamma, const double E_g) {
    constexpr double lambda_h          = 45e-8;
    constexpr double E_threshold       = 0.0;
    constexpr double elementary_charge = 1.0;
    constexpr double E_0               = 4.0e1;

    if (F_ava <= E_threshold) {
        return 0.0;
    } else if (F_ava <= E_0) {
        double a_h                 = 1.582e4;
        double beta                = 0.815009;
        double b_h                 = (beta * E_g) / (lambda_h * elementary_charge);
        double imapct_ionization_h = Gamma * a_h * exp(-Gamma * b_h / F_ava);
        return imapct_ionization_h;
    } else {
        double a_h                 = 6.71e1;
        double beta                = 0.677706;
        double b_h                 = (beta * E_g) / (lambda_h * elementary_charge);
        double imapct_ionization_h = Gamma * a_h * exp(-Gamma * b_h / F_ava);
        return imapct_ionization_h;
    }
}
}  // namespace mcintyre