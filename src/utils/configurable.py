import functools
import inspect
from omegaconf import DictConfig

def configurable(init_func=None, *, from_config=None):
    """
    Decorator to make a function or class's __init__ configurable using Hydra configs.
    
    Can be used in two ways:
    1. As a decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
                
            @classmethod
            def from_config(cls, cfg):
                return {"a": cfg.a, "b": cfg.b}
                
    2. As a decorator with from_config function:
        @configurable(from_config=lambda cfg: {"a": cfg.a, "b": cfg.b})
        def a_func(a, b=2, c=3):
            pass
            
    Args:
        init_func: The __init__ method in usage 1
        from_config: The from_config function in usage 2
    """
    if init_func is not None:
        assert (
            inspect.isfunction(init_func) 
            and from_config is None 
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check documentation for examples."
        
        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")
            
            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)
            
        return wrapped
    else:
        if from_config is None:
            return configurable
        
        assert inspect.isfunction(from_config), "from_config must be a function!"
        
        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            
            wrapped.from_config = from_config
            return wrapped
        
        return wrapper

def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Get explicit arguments from config using from_config function.
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")

    # Check if from_config accepts variable arguments
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )

    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        # Only forward supported arguments
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret

def _called_with_cfg(*args, **kwargs):
    """Check if the function was called with a config object."""
    if len(args) and isinstance(args[0], DictConfig):
        return True
    if isinstance(kwargs.pop("cfg", None), DictConfig):
        return True
    return False