function init_log_file(log_file, use_backbone, inverse_backbone)
    %INIT_LOG_FILE Create CSV header if missing.

    if ~isfile(log_file)
        fid = fopen(log_file, 'w');
        assert(fid ~= -1, 'Cannot open %s for writing.', log_file);

        if use_backbone
            if inverse_backbone
                fprintf(fid, 'ExpID,AUC,TimeElapsed,K,TrainRatio,NonBackboneRatio,Threshold,Precision,Recall,F1Score\n');
            else
                fprintf(fid, 'ExpID,AUC,TimeElapsed,K,TrainRatio,BackboneRatio,Threshold,Precision,Recall,F1Score\n');
            end
        else
            fprintf(fid, 'Iteration,AUC,ElapsedTime,K,TrainRatio,BestThreshold,Precision,Recall,F1Score\n');
        end

        fclose(fid);
    end
end
