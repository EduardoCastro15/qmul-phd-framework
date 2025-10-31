function ensure_nauty_mex()
    % [OPTIMIZATION]
    mexname = ['canonical.' mexext];
    if ~isempty(which(mexname))
        % make sure workers see it too (even if you don't use parfor today)
        try, pctRunOnAll addpath(fileparts(which(mexname))); end
        return
    end

    root = fileparts(mfilename('fullpath'));
    wd   = pwd;
    try
        cd(fullfile(root,'software','nauty26r7'));
        copyfile(fullfile('..','..','canonical.c'), '.','f');
        mex -silent canonical.c nauty.c nautil.c naugraph.c schreier.c naurng.c nausparse.c
        movefile(['canonical.' mexext], fullfile(root,'software'),'f');
    catch ME
        cd(wd); rethrow(ME);
    end
    cd(wd);
    addpath(fullfile(root,'software'));
    try, pctRunOnAll addpath(fullfile(root,'software')); end
end
