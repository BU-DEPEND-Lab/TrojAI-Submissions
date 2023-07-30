import sys

def register_class(cls, name):
    cls.register(name)
    setattr(sys.modules[__name__], name, cls)
    return cls

def register(name):
    """Decorator used to register a learner
    Args:
        name: Name of the learner type to register
    """
    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls