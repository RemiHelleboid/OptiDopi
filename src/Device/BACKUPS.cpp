double device::extract_breakdown_voltage(double brp_threshold) const {
    std::vector<double> list_total_breakdown_probability = get_list_total_breakdown_probability();
    // First we resample the list of voltages and interpolate the list of total breakdown probabilities
    double              min_voltage             = m_list_voltages.front();
    double              max_voltage             = m_list_voltages.back();
    std::size_t         nb_points               = 1000;
    std::vector<double> list_voltages_resampled = utils::linspace(min_voltage, max_voltage, nb_points);
    std::vector<double> list_total_breakdown_probability_resampled =
        Utils::interp1d(m_list_voltages, list_total_breakdown_probability, list_voltages_resampled);
    // Then we find the voltage where the total breakdown probability is above the threshold
    for (std::size_t idx_voltage = 0; idx_voltage < list_voltages_resampled.size(); ++idx_voltage) {
        if (list_total_breakdown_probability_resampled[idx_voltage] > brp_threshold) {
            return list_voltages_resampled[idx_voltage];
        }
    }
    

    return std::numeric_limits<double>::quiet_NaN();
}