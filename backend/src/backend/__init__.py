from .models import ModelService, UnknownModelCheckpointError, register_model


def hello() -> str:
    return "Hello from backend!"


__all__ = ["hello", "ModelService", "UnknownModelCheckpointError", "register_model"]
