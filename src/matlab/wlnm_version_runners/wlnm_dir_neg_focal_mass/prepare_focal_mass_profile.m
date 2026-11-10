function profile = prepare_focal_mass_profile(mass, n)
%PREPARE_FOCAL_MASS_PROFILE Build frozen within-web mass preprocessing.
% Valid masses are finite and strictly positive. Missing standardized masses
% are set to zero (the standardized median) and retained in a separate flag.

    values = NaN(n, 1);
    if isnumeric(mass) && ~isempty(mass)
        raw = double(mass(:));
        upto = min(n, numel(raw));
        values(1:upto) = raw(1:upto);
    end

    valid = isfinite(values) & values > 0;
    log_mass = NaN(n, 1);
    log_mass(valid) = log10(values(valid));

    if any(valid)
        center = median(log_mass(valid));
        quartiles = prctile(log_mass(valid), [25 75]);
        raw_iqr = quartiles(2) - quartiles(1);
    else
        center = 0;
        raw_iqr = 1;
    end
    scale = raw_iqr;
    if ~isfinite(scale) || scale == 0
        scale = 1;
    end

    standardized = zeros(n, 1);
    standardized(valid) = (log_mass(valid) - center) ./ scale;

    profile = struct( ...
        'standardized', standardized, ...
        'missing', ~valid, ...
        'valid_count', double(sum(valid)), ...
        'missing_count', double(sum(~valid)), ...
        'log10_median', double(center), ...
        'log10_iqr', double(raw_iqr), ...
        'scale', double(scale), ...
        'protocol', 'log10_within_web_median_iqr_v1' ...
    );
end
