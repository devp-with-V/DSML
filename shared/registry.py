_METHODS_REGISTRY = {}

def register_method(name):
    """
    Decorator to register a model method under a string name.
    This allows both AutoInsight (via manual selection) and ModelRouter 
    (via LLM JSON output) to call methods from the same shared toolbox.
    """
    def decorator(func):
        if name in _METHODS_REGISTRY:
            raise ValueError(f"Method '{name}' is already registered.")
        _METHODS_REGISTRY[name] = func
        return func
    return decorator

def get_method(name):
    """
    Retrieve a registered method by name.
    """
    if name not in _METHODS_REGISTRY:
        raise ValueError(
            f"Method '{name}' not found. "
            f"Available methods: {list(_METHODS_REGISTRY.keys())}"
        )
    return _METHODS_REGISTRY[name]

def list_methods():
    """List all registered methods."""
    return list(_METHODS_REGISTRY.keys())
