function path = save_dir_neg_trophic_snapshot(folder, empirical, train, pseudo, metadata, metrics)
% Every reconstruction is saved independently of save_confusion/selected-run CSVs.
    if ~isfolder(folder), mkdir(folder); end
    name = regexprep(metadata.Foodweb,'[^a-zA-Z0-9_-]','_');
    name = sprintf('%s_K%d_ratio%.12g_exp%d_seed%d_threshold%.12g_%s.mat', ...
        name,metadata.K,metadata.TrainRatio,metadata.ExperimentID,metadata.Seed, ...
        metadata.Threshold,metadata.Protocol);
    path = fullfile(folder,name);
    if isfile(path), error('WLNM:TrophicSnapshotExists','Refusing to overwrite %s',path); end
    snapshot = struct('empirical',sparse(empirical),'train',sparse(train), ...
        'pseudo',sparse(pseudo),'metadata',metadata,'metrics',metrics);
    temporary = [tempname(folder) '.mat']; % Does not consume MATLAB's RNG stream.
    cleanup = onCleanup(@() remove_temporary(temporary)); %#ok<NASGU>
    save(temporary,'snapshot','-v7');
    [ok,msg] = movefile(temporary,path);
    if ~ok, error('WLNM:TrophicSnapshotWrite','%s',msg); end
end
function remove_temporary(path)
    if isfile(path), delete(path); end
end
