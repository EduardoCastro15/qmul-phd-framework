function protocol = resolve_negative_sampling_protocol( ...
        eligibility_mode, negative_positive_ratio, sampling_strategy, ...
        topup_policy, legacy_use_role_filter, legacy_mass_enabled)
%RESOLVE_NEGATIVE_SAMPLING_PROTOCOL Validate and normalize sampler settings.
%
% A non-empty eligibility_mode is canonical. The legacy booleans are used
% only when callers omit that mode, preserving historical role-or-mass and
% k-fold calls while making new WLNM_dir_neg experiments fully explicit.

    if nargin < 1, eligibility_mode = ''; end
    if nargin < 2 || isempty(negative_positive_ratio), negative_positive_ratio = 2; end
    if nargin < 3 || isempty(sampling_strategy)
        sampling_strategy = 'uniform_without_replacement';
    end
    if nargin < 4 || isempty(topup_policy)
        topup_policy = 'uniform_remaining_nonlinks';
    end
    if nargin < 5 || isempty(legacy_use_role_filter), legacy_use_role_filter = true; end
    if nargin < 6 || isempty(legacy_mass_enabled), legacy_mass_enabled = false; end

    mode = normalize_option(eligibility_mode);
    if isempty(mode)
        if logical(legacy_use_role_filter) && logical(legacy_mass_enabled)
            mode = 'role_or_mass';
        elseif logical(legacy_use_role_filter)
            mode = 'role_only';
        elseif logical(legacy_mass_enabled)
            mode = 'mass_only';
        else
            mode = 'all_nonlinks';
        end
    end

    allowed_modes = {'role_only', 'role_or_mass', 'mass_only', 'all_nonlinks'};
    if ~any(strcmp(mode, allowed_modes))
        error('resolve_negative_sampling_protocol:InvalidEligibilityMode', ...
            'Eligibility mode must be one of: %s. Got "%s".', ...
            strjoin(allowed_modes, ', '), char(string(eligibility_mode)));
    end

    ratio = double(negative_positive_ratio);
    if ~isscalar(ratio) || ~isfinite(ratio) || ratio <= 0
        error('resolve_negative_sampling_protocol:InvalidNegativePositiveRatio', ...
            'Negative-positive ratio must be one positive finite scalar.');
    end

    strategy = normalize_option(sampling_strategy);
    if strcmp(strategy, 'random_eligible_pool')
        strategy = 'uniform_without_replacement';
    end
    if ~strcmp(strategy, 'uniform_without_replacement')
        error('resolve_negative_sampling_protocol:InvalidSamplingStrategy', ...
            'Only uniform_without_replacement is currently supported. Got "%s".', ...
            char(string(sampling_strategy)));
    end

    normalized_topup = normalize_option(topup_policy);
    if any(strcmp(normalized_topup, {'random', 'random_topup', 'uniform_random'}))
        normalized_topup = 'uniform_remaining_nonlinks';
    end
    if ~any(strcmp(normalized_topup, {'uniform_remaining_nonlinks', 'error'}))
        error('resolve_negative_sampling_protocol:InvalidTopupPolicy', ...
            'Top-up policy must be uniform_remaining_nonlinks or error. Got "%s".', ...
            char(string(topup_policy)));
    end

    protocol = struct();
    protocol.eligibility_mode = mode;
    protocol.negative_positive_ratio = ratio;
    protocol.sampling_strategy = strategy;
    protocol.topup_policy = normalized_topup;
    protocol.use_role_filter = any(strcmp(mode, {'role_only', 'role_or_mass'}));
    protocol.use_mass_constraint = any(strcmp(mode, {'mass_only', 'role_or_mass'}));
end

function value = normalize_option(value)
    value = lower(strtrim(char(string(value))));
    value = strrep(value, '-', '_');
    value = regexprep(value, '\s+', '_');
end
