function registry = get_version_registry()
    %GET_VERSION_REGISTRY Map version keys to runner function handles.
    % Keys are lowercased strings (e.g., 'wlnm_dir_neg').

    registry = containers.Map('KeyType','char','ValueType','any');

    % Register each version here (one line per version):
    registry('wlnm_original') = @run_wlnm_original;
    registry('wlnm_directed') = @run_wlnm_directed;
    registry('wlnm_negative') = @run_wlnm_negative;
    registry('wlnm_dir_neg')  = @run_wlnm_dir_neg;

    % Examples to add later:
    % registry('cn_baseline')      = @run_cn_baseline;
    % registry('path_based')       = @run_path_based;
    % registry('rw_based')         = @run_rw_based;
    % registry('latent_features')  = @run_latent_features;
    % registry('sbm')              = @run_sbm;
end
