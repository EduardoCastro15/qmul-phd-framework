function outputDir = run_wlnm_dir_neg_trophic_pilot(outputDir, varargin)
% Isolated opt-in rerun: PlotB/Ythan, 60%, 100 runs, original negative protocol.
% Example: run_wlnm_dir_neg_trophic_pilot('/absolute/path/to/new/pilot')
% Smoke check: ...(...,'ExperimentIDs',1) (not sufficient for Tukey retention).
    p = inputParser;
    addParameter(p,'ExperimentIDs',1:100,@(x) isnumeric(x) && isvector(x) && ~isempty(x));
    addParameter(p,'HighPrecision','auto');
    parse(p,varargin{:});
    ids = p.Results.ExperimentIDs;
    validateattributes(ids,{'numeric'},{'integer','finite','>=',1,'<=',100});
    if numel(unique(ids)) ~= numel(ids), error('WLNM:TrophicConfiguration','Duplicate experiment IDs.'); end
    modelDir = fileparts(mfilename('fullpath'));
    matlabDir = fileparts(fileparts(modelDir));
    if nargin < 1 || isempty(outputDir)
        outputDir = fullfile(matlabDir,'data','result_wlnm_dir_neg_roleonly_train60_trophic_v2_pilot');
    end
    outputDir = char(java.io.File(outputDir).getCanonicalPath());
    if isfolder(outputDir) || isfile(outputDir)
        error('WLNM:TrophicOutputExists','Use a NEW output directory; refusing to overwrite %s.',outputDir);
    end
    names = {'Dutch Microfauna food web PlotB_tax_mass','Ythan Estuary_tax_mass'};
    for k=1:2
        validate_dir_neg_trophic_scope(names{k},'WLNM_dir_neg','validated_v2',p.Results.HighPrecision);
    end
    oldPath = path; restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
    oldRng = rng; restoreRng = onCleanup(@() rng(oldRng)); %#ok<NASGU>
    for folder = {'wlnm_version_runners','software','metrics','logging'}
        addpath(genpath(fullfile(matlabDir,folder{1})));
    end
    config = struct('version','WLNM_dir_neg','useParallel',false, ...
        'use_backbone',false,'inverse_backbone',false,'sweepBackboneTrain',false, ...
        'BackboneRatio',0.5,'backboneRatioRange',0.1:0.1:0.9,'numExperiments',100, ...
        'experimentIDList',ids,'baseSeed',12345,'resampleSplitsEachExperiment',true, ...
        'nodeSelection','random','checkConnectivity',false,'adaptiveConnectivity',false, ...
        'cvSaveConfusion',false,'exportAuxiliaryCSVs',false,'evaluate_on_all_unseen',false, ...
        'exportBackboneCSV',false,'thresholdMode','fixed','fixedThreshold',0.5, ...
        'thresholdSweepEnabled',false,'thresholdSweepRange',0.1:0.1:0.9, ...
        'negativeEligibilityMode','role_only','negativePositiveRatio',2, ...
        'negativeSamplingStrategy','uniform_without_replacement', ...
        'negativeTopupPolicy','uniform_remaining_nonlinks', ...
        'negativeMassEligibilityEnabled',false,'negativeMassEligibilityThreshold',1, ...
        'useGraphEncodingParallel',false,'computeEcologicalMetrics',true, ...
        'trophicLevelProtocol','validated_v2','trophicHighPrecision',p.Results.HighPrecision, ...
        'trophicSnapshotDir',fullfile(outputDir,'ecological_snapshots'), ...
        'artifactDir',fullfile(outputDir,'confusion_matrix_csv'));
    manifest = struct('protocol','validated_v2','status','started', ...
        'config',config,'K',10,'TrainRatio',0.6,'foodwebs',{names}, ...
        'matlabVersion',version,'toolboxes',ver,'createdUTC', ...
        char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ssXXX')));
    manifest.codeHashes = source_hashes(matlabDir);
    inputs = cell(2,1);
    for k=1:2
        source = fullfile(matlabDir,'data','foodwebs_mat',[names{k} '.mat']);
        inputs{k} = load(source,'net','taxonomy','mass','role');
        inputs{k}.dataname = names{k}; inputs{k}.backbone_mask = [];
        manifest.inputs(k) = struct('path',source,'sha256',sha256_file(source));
    end
    mkdir(outputDir); mkdir(fullfile(outputDir,'prediction_scores_logs'));
    save(fullfile(outputDir,'pilot_manifest.mat'),'manifest','-v7');
    write_json(fullfile(outputDir,'pilot_manifest.json'),manifest);
    fid = fopen(fullfile(outputDir,'RUN_MANIFEST.txt'),'w');
    closer = onCleanup(@() fclose(fid));
    fprintf(fid,['Version=WLNM_dir_neg\nNumExperiments=100\nK=10\nTrainRatio=60\n' ...
        'Threshold=0.5\nTrophicLevelProtocol=validated_v2\nNegativeEligibilityMode=role_only\n' ...
        'NegativePositiveRatio=2\nNegativeSamplingStrategy=uniform_without_replacement\n' ...
        'NegativeTopupPolicy=uniform_remaining_nonlinks\n']);
    clear closer;
    try
        for k=1:2
            data = inputs{k};
            rows = run_wlnm_dir_neg(data,10,0.6,config);
            T = struct2table(rows);
            T.TrainRatio = 100*T.TrainRatio; % Existing result CSV convention.
            T.Foodweb = repmat(string(names{k}),height(T),1);
            writetable(T,fullfile(outputDir,'prediction_scores_logs', ...
                [names{k} '_results_random_wlnm_dir_neg.csv']));
            save(fullfile(outputDir,[names{k} '_results.mat']),'rows','-v7');
        end
        manifest.status = 'completed';
    catch err
        manifest.status = 'failed'; manifest.error = err.message;
        write_json(fullfile(outputDir,'pilot_manifest.json'),manifest);
        rethrow(err);
    end
    write_json(fullfile(outputDir,'pilot_manifest.json'),manifest);
    save(fullfile(outputDir,'pilot_manifest.mat'),'manifest','-v7');
end

function records = source_hashes(root)
    records = struct('path',{},'sha256',{});
    entries = dir(root);
    for k=1:numel(entries)
        e = entries(k);
        if startsWith(e.name,'.') || strcmp(e.name,'data'), continue; end
        path = fullfile(root,e.name);
        if e.isdir
            records = [records source_hashes(path)]; %#ok<AGROW>
        elseif endsWith(e.name,{'.m','.c','.cpp',['.' mexext]})
            records(end+1) = struct('path',path,'sha256',sha256_file(path)); %#ok<AGROW>
        end
    end
end

function hash = sha256_file(path)
    fid = fopen(path,'rb');
    if fid < 0, error('WLNM:TrophicInput','Cannot open %s',path); end
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    bytes = fread(fid,Inf,'*uint8');
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    hash = lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
end

function write_json(path,value)
    fid = fopen(path,'w');
    if fid < 0, error('WLNM:TrophicOutput','Cannot write %s',path); end
    closer = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end
