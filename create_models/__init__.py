import os
import functools
from . import base

_TF_KERAS_FRAMEWORK_NAME = 'tf.keras'

_DEFAULT_KERAS_FRAMEWORK = _TF_KERAS_FRAMEWORK_NAME
_KERAS_FRAMEWORK = None
_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None
_KERAS_LOSSES = None


def inject_global_losses(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['losses'] = _KERAS_LOSSES
        return func(*args, **kwargs)

    return wrapper


def inject_global_submodules(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = _KERAS_BACKEND
        kwargs['layers'] = _KERAS_LAYERS
        kwargs['models'] = _KERAS_MODELS
        kwargs['utils'] = _KERAS_UTILS
        return func(*args, **kwargs)

    return wrapper


def filter_kwargs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}
        return func(*args, **new_kwargs)

    return wrapper


def framework():
    """Return name of Segmentation Models framework"""
    return _KERAS_FRAMEWORK


def set_framework(name):
    """Set framework for Segmentation Models

    Args:
        name (str): one of ``keras``, ``tf.keras``, case insensitive.

    Raises:
        ValueError: in case of incorrect framework name.
        ImportError: in case framework is not installed.

    """
    name = name.lower()

    if name == _TF_KERAS_FRAMEWORK_NAME:
        from tensorflow import keras
        import backbones.tfkeras  # init custom objects
    else:
        raise ValueError('Not correct module name `{}`, use `{}'.format(
            name, _TF_KERAS_FRAMEWORK_NAME))

    global _KERAS_BACKEND, _KERAS_LAYERS, _KERAS_MODELS
    global _KERAS_UTILS, _KERAS_LOSSES, _KERAS_FRAMEWORK

    _KERAS_FRAMEWORK = name
    _KERAS_BACKEND = keras.backend
    _KERAS_LAYERS = keras.layers
    _KERAS_MODELS = keras.models
    _KERAS_UTILS = keras.utils
    _KERAS_LOSSES = keras.losses

    # allow losses/metrics get keras submodules
    base.KerasObject.set_submodules(
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
    )


# set default framework
_framework = os.environ.get('SM_FRAMEWORK', _DEFAULT_KERAS_FRAMEWORK)
set_framework(_framework)

print('Models: using `{}` framework.'.format(_KERAS_FRAMEWORK))

# import helper modules
# from . import losses
# from . import metrics
# from . import utils

# wrap segmentation models with framework modules
from .backbones.backbones_factory import Backbones
from .models.unet import Unet as _Unet

Unet = inject_global_submodules(_Unet)
# get_available_backbone_names = Backbones.models_names


def get_preprocessing(name):
    preprocess_input = Backbones.get_preprocessing(name)
    preprocess_input = inject_global_submodules(preprocess_input)
    preprocess_input = filter_kwargs(preprocess_input)
    return preprocess_input


__all__ = [
    'Unet',
    'set_framework', 'framework',
    'get_preprocessing', 'get_available_backbone_names',
]
