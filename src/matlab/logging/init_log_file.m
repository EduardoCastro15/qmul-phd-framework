function init_log_file(log_file)
    %INIT_LOG_FILE Create CSV header if missing.

    if ~isfile(log_file)
        fid = fopen(log_file, 'w');
        assert(fid ~= -1, 'Cannot open %s for writing.', log_file);
        fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,BestThreshold,Precision,Recall,F1Score\n');
        fclose(fid);
    end
end
