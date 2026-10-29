"""Define utilities for loading user-provided ML models."""

import json
import importlib

__author__ = "Ruiqi Luo"
__status__ = "Development"
__date__ = "2026_10_29"
__version__ = "0.0.5"


def validate_model_spec(spec):
    """Check that ``spec`` is a well-formed, JSON-serializable model specification, without importing anything.

    Args:
        spec (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.

    Raises:
        ValueError: If ``spec`` is missing, malformed or not JSON serializable.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            'model must be a dict like {"entry": "package.module:ClassName", '
            '"kwargs": {...}}'
        )
    entry = spec.get("entry")
    if not isinstance(entry, str) or entry.count(":") != 1 or not all(entry.split(":")):
        raise ValueError(
            f'model["entry"] must be of the form "package.module:ClassName", got '
            f"{entry!r}"
        )
    if not isinstance(spec.get("kwargs", {}), dict):
        raise ValueError('model["kwargs"] must be a dict')
    try:
        json.dumps(spec)
    except TypeError as e:
        raise ValueError(f"model must be JSON serializable (FireWorks stores it): {e}")


def load_model(spec):
    """Instantiate the user's model from its specification.

    .. warning::
        Loading executes user code: the module in ``entry`` is imported and the
        class is instantiated with ``kwargs``. The module must be importable on
        the machine that runs the Firework, not only on the one that builds the
        workflow.

    Args:
        spec (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.

    Returns:
        object: The model instance; it has a callable ``predict``.

    Raises:
        TypeError: If the instance has no callable ``predict``.
    """
    validate_model_spec(spec)
    module_name, class_name = spec["entry"].split(":")
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class(**spec.get("kwargs", {}))
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"{spec['entry']} instance has no callable `predict` method")
    return model
