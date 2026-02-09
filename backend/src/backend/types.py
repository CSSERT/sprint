from typing import Any, NotRequired, TypedDict

import torch
import torch.nn as nn

ModelKwargs = dict[str, Any]
ModelRegistry = dict[str, type[nn.Module]]


class ModelConfig(TypedDict, total=False):
    model_name: str
    model_kwargs: ModelKwargs


class Checkpoint(TypedDict):
    config: ModelConfig
    state_dict: dict[str, torch.Tensor]
    metadata: NotRequired[dict[str, Any]]
