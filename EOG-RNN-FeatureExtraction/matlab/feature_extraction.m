function combined_features = feature_extraction(hEOG, vEOG)
    % Time-domain features
    hEOG_time_features = arrayfun(@(row) extract_time_domain_features(hEOG(row, :)), 1:size(hEOG, 1), 'UniformOutput', false);
    vEOG_time_features = arrayfun(@(row) extract_time_domain_features(vEOG(row, :)), 1:size(vEOG, 1), 'UniformOutput', false);

    hEOG_time_features = cell2mat(hEOG_time_features);
    vEOG_time_features = cell2mat(vEOG_time_features);

    % Frequency-domain features
    hEOG_freq_features = arrayfun(@(row) extract_frequency_domain_features(hEOG(row, :)), 1:size(hEOG, 1), 'UniformOutput', false);
    vEOG_freq_features = arrayfun(@(row) extract_frequency_domain_features(vEOG(row, :)), 1:size(vEOG, 1), 'UniformOutput', false);

    hEOG_freq_features = cell2mat(hEOG_freq_features);
    vEOG_freq_features = cell2mat(vEOG_freq_features);

    % Combine features
    combined_features = [hEOG_time_features, vEOG_time_features, hEOG_freq_features, vEOG_freq_features];
end

function features = extract_time_domain_features(signal)
    features.signal_variance = var(signal);
    features.signal_iqr = iqr(signal);
    features.mean = mean(signal);
    features.std = std(signal);
    features.max = max(signal);
    features.min = min(signal);
    features.skew = skewness(signal);
    features.kurtosis = kurtosis(signal);
    features.energy = sum(signal.^2);
    features.median = median(signal);
    features.entropy = entropy(signal);  % You need to implement or find an entropy function

    % Calculate the slope of the signal
    if length(signal) > 1
        signal_slope = diff(signal);  % First derivative (slope)
        features.mean_slope = mean(signal_slope);
        features.std_slope = std(signal_slope);
    else
        features.mean_slope = 0;
        features.std_slope = 0;
    end
end

function features = extract_frequency_domain_features(signal)
    fft_vals = fft(signal);
    fft_magnitude = abs(fft_vals);
    power = fft_magnitude.^2;

    features.fft_max_freq = find(fft_magnitude == max(fft_magnitude)) - 1;  % MATLAB index starts from 1
    features.fft_mean_power = mean(power);
    features.fft_peak_power = max(power);
end
