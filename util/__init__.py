import os
import pkgutil
import importlib

__all__ = []

package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module  
    __all__.append(module_name)

    for attr in dir(module):
        if not attr.startswith("_"):  
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
