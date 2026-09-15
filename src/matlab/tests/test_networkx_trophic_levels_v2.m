function tests = test_networkx_trophic_levels_v2
    tests = functiontests(localfunctions);
end
function setupOnce(tc)
    base = fileparts(fileparts(mfilename('fullpath')));
    tc.TestData.oldPath = path;
    addpath(fullfile(base,'wlnm_version_runners','wlnm_dir_neg'),fullfile(base,'metrics'));
end
function teardownOnce(tc)
    path(tc.TestData.oldPath);
end
function testChainAndSingleton(tc)
    A = sparse([1 2],[2 3],1,3,3);
    [t,s] = compute_networkx_trophic_levels_v2(A,'off');
    verifyEqual(tc,t,[1;2;3],'AbsTol',1e-12);
    verifyEqual(tc,s.StatusCode,0); verifyEqual(tc,s.NumBasal,1);
    verifyEqual(tc,compute_networkx_trophic_levels_v2(sparse(1), 'off'),1);
end
function testEmpty(tc)
    [t,s] = compute_networkx_trophic_levels_v2(sparse(0,0),'off');
    verifyEmpty(tc,t); verifyEqual(tc,s.StatusCode,1);
end
function testUnfedCycle(tc)
    [t,s] = compute_networkx_trophic_levels_v2(sparse([1 2],[2 1],1,2,2),'off');
    verifyTrue(tc,all(isnan(t))); verifyEqual(tc,s.StatusCode,6);
end
function testBasalUnreachable(tc)
    % 1 feeds 2; unfed 3<->4 also feeds 2. One weak component, mixed reachability.
    A = sparse([1 3 4 3],[2 4 3 2],1,4,4);
    [t,s] = compute_networkx_trophic_levels_v2(A,'off');
    verifyTrue(tc,all(isnan(t))); verifyEqual(tc,s.StatusCode,7);
    verifyEqual(tc,s.NumUnreachable,2); verifyEqual(tc,s.NumSpeciesWithLevel,0);
end
function testStableAboveLegacyCap(tc)
    A = sparse(6,6); A(2:6,2:6) = ones(5)-eye(5); A(1,2)=1;
    [t,s] = compute_networkx_trophic_levels_v2(A,'off');
    verifyEqual(tc,t,[1;22;26;26;26;26],'AbsTol',1e-10);
    verifyEqual(tc,s.StatusCode,0); verifyTrue(tc,s.AboveLegacyCap);
    v2 = compute_dir_neg_foodweb_metrics(A,'validated_v2','off');
    verifyEqual(tc,v2.TrophicV2Diagnostics.LegacyStatusCode,3);
    verifyTrue(tc,isnan(v2.TrophicV2Diagnostics.LegacyMean));
    verifyEqual(tc,v2.NetworkXMeanTrophicLevel,127/6,'AbsTol',1e-10);
end
function testDisconnectedTieAndLoopRemoval(tc)
    A = sparse([1 3 1],[2 4 1],1,5,5);
    [t,s] = compute_networkx_trophic_levels_v2(A,'off');
    verifyEqual(tc,t(1:2),[1;2]); verifyTrue(tc,all(isnan(t(3:5))));
    verifyEqual(tc,s.NumSpeciesLargest,2); verifyEqual(tc,s.LargestFraction,0.4);
end
function testUniqueLccPermutation(tc)
    A = sparse([1 2],[2 3],1,4,4); p=[3 4 1 2];
    t = compute_networkx_trophic_levels_v2(A,'off');
    reordered = compute_networkx_trophic_levels_v2(A(p,p),'off');
    verifyEqual(tc,reordered,t(p));
end
function testLegacyAndOtherMetricsUnchanged(tc)
    A = sparse([1 2],[2 3],1,3,3);
    legacy = compute_foodweb_metrics(A);
    verifyEqual(tc,compute_dir_neg_foodweb_metrics(A,'legacy_v1','off'),legacy);
    v2 = compute_dir_neg_foodweb_metrics(A,'validated_v2','off');
    names = fieldnames(legacy);
    for k=1:numel(names)
        if ~startsWith(names{k},'NetworkX')
            verifyEqual(tc,v2.(names{k}),legacy.(names{k}));
        end
    end
end
function testScope(tc)
    validate_dir_neg_trophic_scope('other','WLNM_dir_neg','validated_v2','off');
    validate_dir_neg_trophic_scope('other','WLNM_dir_neg_kfold','validated_v2','off');
    verifyError(tc,@() validate_dir_neg_trophic_scope('Ythan Estuary_tax_mass','WLNM_original','validated_v2','off'),'WLNM:TrophicScope');
    verifyError(tc,@() validate_dir_neg_trophic_scope('Ythan Estuary_tax_mass','WLNM_dir_neg','validated','off'),'WLNM:TrophicConfiguration');
    validate_dir_neg_trophic_scope('any','WLNM_original','legacy_v1','off');
end
function testIllConditionedReachable(tc)
    % Waiting for 38 consecutive forward choices; reset edges give very high TL.
    % Resource orientation is the transpose of the consumer's diet transition.
    n=40; A=sparse(n,n);
    for j=2:n-1, A(j+1,j)=1; A(2,j)=1; end
    A(1,n)=1; A(2,n)=1;
    [t,s]=compute_networkx_trophic_levels_v2(A,'off');
    verifyLessThanOrEqual(tc,s.ReciprocalCondition,1e-10);
    verifyEqual(tc,s.NumUnreachable,0); verifyTrue(tc,all(isnan(t)));
    verifyEqual(tc,s.StatusCode,2);
    if exist('sym','file')==2 && license('test','Symbolic_Toolbox')
        oldDigits=digits;
        [t,s]=compute_networkx_trophic_levels_v2(A,'required');
        verifyEqual(tc,s.StatusCode,0); verifyEqual(tc,s.PrecisionDigits,100);
        verifyTrue(tc,all(isfinite(t))); verifyEqual(tc,digits,oldDigits);
    else
        verifyError(tc,@() compute_networkx_trophic_levels_v2(A,'required'),'WLNM:TrophicPrecisionUnavailable');
    end
end
function testSnapshotRoundTripAndRng(tc)
    folder=tempname; mkdir(folder); cleanup=onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
    A=sparse([1 2],[2 3],1,3,3);
    metadata=struct('Foodweb','Ythan Estuary_tax_mass','K',10,'TrainRatio',0.6, ...
        'ExperimentID',1,'Seed',42,'Threshold',0.5,'Protocol','validated_v2','NodeIDs',(1:3)');
    before=rng;
    metrics=compute_dir_neg_foodweb_metrics(A,'validated_v2','off');
    filename=save_dir_neg_trophic_snapshot(folder,A,A,A,metadata,metrics);
    verifyEqual(tc,rng,before);
    stored=load(filename);
    verifyEqual(tc,stored.snapshot.pseudo,A); verifyTrue(tc,issparse(stored.snapshot.pseudo));
    verifyEqual(tc,stored.snapshot.metadata,metadata);
    verifyEqual(tc,stored.snapshot.metrics,metrics);
    verifyError(tc,@() save_dir_neg_trophic_snapshot(folder,A,A,A,metadata,metrics),'WLNM:TrophicSnapshotExists');
end
