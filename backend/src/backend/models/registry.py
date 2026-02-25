from ..types.model import QuantileModel

ModelRegistry = dict[str, type[QuantileModel]]

_MODEL_REGISTRY: ModelRegistry = {}


def register_model(name: str):
    # Decorator pattern to register model!
    def decorator(cls: type[QuantileModel]) -> type[QuantileModel]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_registry() -> ModelRegistry:
    return _MODEL_REGISTRY
