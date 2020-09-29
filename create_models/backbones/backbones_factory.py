import copy
from . import model as eff
from .models_factory import ModelsFactory


class BackbonesFactory(ModelsFactory):
    _default_feature_layers = {

        # List of layers to take features from backbone in the following order:
        # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
        # resolution (Height x Width) than input image.

        # EfficientNets
        'efficientnetb0': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb1': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb2': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb3': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb4': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb5': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb6': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb7': ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),

    }

    _models_update = {

        'efficientnetb0': [eff.EfficientNetB0, eff.preprocess_input],
        'efficientnetb1': [eff.EfficientNetB1, eff.preprocess_input],
        'efficientnetb2': [eff.EfficientNetB2, eff.preprocess_input],
        'efficientnetb3': [eff.EfficientNetB3, eff.preprocess_input],
        'efficientnetb4': [eff.EfficientNetB4, eff.preprocess_input],
        'efficientnetb5': [eff.EfficientNetB5, eff.preprocess_input],
        'efficientnetb6': [eff.EfficientNetB6, eff.preprocess_input],
        'efficientnetb7': [eff.EfficientNetB7, eff.preprocess_input],
    }

    @property
    def models(self):
        all_models = copy.copy(self._models)
        all_models.update(self._models_update)
        return all_models

    def get_backbone(self, name, *args, **kwargs):
        model_fn, _ = self.get(name)
        model = model_fn(*args, **kwargs)
        return model

    def get_feature_layers(self, name, n=5):
        return self._default_feature_layers[name][:n]

    def get_preprocessing(self, name):
        return self.get(name)[1]


Backbones = BackbonesFactory()
