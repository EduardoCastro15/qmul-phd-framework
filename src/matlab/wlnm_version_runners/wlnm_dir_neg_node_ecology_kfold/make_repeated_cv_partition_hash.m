function hash_text = make_repeated_cv_partition_hash(folds)
%MAKE_REPEATED_CV_PARTITION_HASH SHA-256 for a complete fold assignment.

    payload_rows = sortrows( ...
        [double(folds.i(:)), double(folds.j(:)), double(folds.fold_id(:))], ...
        [1 2]);
    payload = sprintf('n=%d|k=%d|edges=', folds.n, folds.k);
    if ~isempty(payload_rows)
        payload = [payload, sprintf('%d:%d:%d;', payload_rows')];
    end
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(uint8(payload));
    bytes = typecast(digest.digest(), 'uint8');
    hash_text = lower(reshape(dec2hex(bytes, 2)', 1, []));
end
