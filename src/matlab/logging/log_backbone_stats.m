function log_backbone_stats(csv_path, dataname, net, st)
    %LOG_BACKBONE_STATS  Append per-foodweb backbone overview row to CSV.
    %
    % csv_path : e.g. 'data/result/backbone_stats/backbone_overview_per_foodweb.csv'
    % dataname : foodweb name (string)
    % net      : adjacency (used only to get n and total edges)
    % st       : struct from backbone_regime with fields:
    %            m_edges, kept, fraction, method, q_start, q_used, thr

    % ---- Ensure output folder exists ----
    [folder, ~, ~] = fileparts(csv_path);
    if ~isempty(folder) && ~exist(folder, 'dir')
        mkdir(folder);
    end

    % ---- Open file for appending ----
    file_exists = isfile(csv_path);
    fid = fopen(csv_path, 'a');
    if fid == -1
        error('log_backbone_stats:IOError', ...
              'Could not open %s for writing.', csv_path);
    end
    c = onCleanup(@() fclose(fid));

    % ---- Write header if new file ----
    if ~file_exists
        fprintf(fid, ['Foodweb,NumNodes,TotalEdges,BackboneEdges,', ...
                      'BackboneFraction,Method,q_start,q_used,thr\n']);
    end

    % ---- Basic counts ----
    n = size(net, 1);
    if isfield(st, 'm_edges')
        m_total = st.m_edges;
    else
        m_total = nnz(net);
    end

    if isfield(st, 'kept')
        m_backbone = st.kept;
    else
        m_backbone = round(st.fraction * m_total);
    end

    % ---- Safe foodweb name (escape quotes, wrap with ") ----
    % Replace any " by "" (CSV escaping), then wrap in double quotes.
    name_safe = strrep(dataname, '"', '""');

    % ---- Write row ----
    fprintf(fid, '"%s",%d,%d,%d,%.6f,%s,%.4f,%.4f,%.6g\n', ...
            name_safe, ...
            n, ...
            m_total, ...
            m_backbone, ...
            st.fraction, ...
            st.method, ...
            st.q_start, ...
            st.q_used, ...
            st.thr);
end
