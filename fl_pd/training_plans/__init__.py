try:
    import fedbiomed
except ImportError:
    raise ImportError(
        "The fedbiomed package is not installed in this environment. "
        "It has dependencies that conflict with pcntoolkit, so it should be installed "
        "in a separate environment"
    )
