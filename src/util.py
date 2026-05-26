from typing import Any


def obj_dot_notation_set(key: str, obj: object, value: Any) -> object:
    root = obj
    parts = key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return root


def dict_dot_notation_set(key: str, obj: dict, value: Any) -> dict:
    parts = key.split(".")
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value
    return obj
