import sys
import logging
logger = logging.getLogger(__name__)

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
        logger.info("Register {}".format(name))
        name = name.lower()
        return lambda c: register_class(c, name)
    else:
        cls = name
        name = cls.__name__
        logger.info("Register class {}".format(name))
        register_class(cls, name.lower())

    return cls