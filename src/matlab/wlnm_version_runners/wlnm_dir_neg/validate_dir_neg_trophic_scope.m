function validate_dir_neg_trophic_scope(~, version, protocol, highPrecision)
% Validate the shared trophic protocol for both directed-negative runners.
    if ~any(strcmp(protocol,{'legacy_v1','validated_v2'})) || ...
            ~any(strcmp(highPrecision,{'auto','off','required'}))
        error('WLNM:TrophicConfiguration','Unknown trophic protocol or precision mode.');
    end
    if strcmp(protocol,'legacy_v1'), return; end
    allowed_versions = {'WLNM_dir_neg','WLNM_dir_neg_kfold'};
    if ~any(strcmpi(version,allowed_versions))
        error('WLNM:TrophicScope', ...
            'validated_v2 is available only for WLNM_dir_neg and WLNM_dir_neg_kfold.');
    end
    if strcmp(highPrecision,'required') && ...
            ~(exist('sym','file') == 2 && license('test','Symbolic_Toolbox'))
        error('WLNM:TrophicPrecisionUnavailable','Symbolic Math Toolbox is required.');
    end
end
