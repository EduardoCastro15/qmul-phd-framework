function init_log_file(log_file, sweepBackboneTrain)
    %INIT_LOG_FILE Create CSV header if missing.

    if ~isfile(log_file)
        fid = fopen(log_file, 'w');
        assert(fid ~= -1, 'Cannot open %s for writing.', log_file);

        if sweepBackboneTrain
            fprintf(fid, 'ExpID,AUC,TimeElapsed,K,TrainRatio,BackboneRatio,Threshold,Precision,Recall,F1Score\n');
        else
            fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,BestThreshold,Precision,Recall,F1Score\n');
        end

        fclose(fid);
    end
end
