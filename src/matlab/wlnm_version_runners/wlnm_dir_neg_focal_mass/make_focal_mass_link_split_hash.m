function hash_text = make_focal_mass_link_split_hash(n, train_links, test_links)
%MAKE_FOCAL_MASS_LINK_SPLIT_HASH SHA-256 for ordered train/test link sets.

    train_links = sortrows(double(train_links), [1 2]);
    test_links = sortrows(double(test_links), [1 2]);
    payload = sprintf('n=%d|train=', n);
    if ~isempty(train_links)
        payload = [payload, sprintf('%d:%d;', train_links')]; %#ok<AGROW>
    end
    payload = [payload, '|test=']; %#ok<AGROW>
    if ~isempty(test_links)
        payload = [payload, sprintf('%d:%d;', test_links')]; %#ok<AGROW>
    end
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(uint8(payload));
    bytes = typecast(digest.digest(), 'uint8');
    hash_text = lower(reshape(dec2hex(bytes, 2)', 1, []));
end
