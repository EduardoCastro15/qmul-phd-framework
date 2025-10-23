function runner = resolve_runner(registry, versionName)
    %RESOLVE_RUNNER Return the runner handle for the requested version.

    key = lower(string(versionName));
    if ~isKey(registry, key)
        available = strjoin(keys(registry), ', ');
        error('Version "%s" not registered. Available versions: %s', key, available);
    end
    runner = registry(key);
end
