"""Keep routing dependencies isolated from the dedicated flight environment."""

def pytest_ignore_collect(collection_path, config):
    if config.option.markexpr.strip() == 'gpu' and collection_path.name.startswith('test_'):
        return collection_path.name != 'test_gpu.py'
    return None
