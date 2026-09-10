import logging
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    total_params: int
    trainable_params: int
    flops: int
    total_params_m: float
    trainable_params_m: float
    gflops: float


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def _module_flops(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return 0

    input_tensor = inputs[0]
    output_tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(output_tensor, torch.Tensor):
        return 0

    if isinstance(module, nn.Conv2d):
        batch_size = output_tensor.shape[0]
        output_channels = output_tensor.shape[1]
        output_height = output_tensor.shape[2]
        output_width = output_tensor.shape[3]
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        conv_per_position_flops = kernel_height * kernel_width * in_channels * output_channels // groups
        active_positions = batch_size * output_height * output_width
        bias_flops = output_channels * active_positions if module.bias is not None else 0
        return int(conv_per_position_flops * active_positions + bias_flops)

    if isinstance(module, nn.Linear):
        output_elements = output_tensor.numel()
        bias_flops = output_elements if module.bias is not None else 0
        return int(module.in_features * output_elements + bias_flops)

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return int(input_tensor.numel())

    if isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        return int(output_tensor.numel())

    return 0


def _hook_flops(model: nn.Module, example_input: torch.Tensor) -> int:
    handles = []
    flops = 0

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        flops += _module_flops(module, inputs, output)

    profiled_types: Iterable[type[nn.Module]] = (
        nn.Conv2d,
        nn.Linear,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
    )

    for module in model.modules():
        if isinstance(module, profiled_types):
            handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for handle in handles:
            handle.remove()

    return flops


def estimate_flops(model: nn.Module, example_input: torch.Tensor) -> int:
    try:
        with torch.no_grad(), torch.profiler.profile(with_flops=True) as profiler:
            model(example_input)
        profiler_flops = sum(event.flops for event in profiler.key_averages() if event.flops is not None)
        if profiler_flops > 0:
            return int(profiler_flops / 2)
    except Exception as exc:
        logger.warning(f"torch.profiler FLOPs estimation failed, falling back to module hooks: {exc}")

    return _hook_flops(model, example_input)


def profile_model(model: nn.Module, example_input: torch.Tensor) -> ModelProfile:
    was_training = model.training
    model.eval()

    try:
        total_params, trainable_params = count_parameters(model)
        flops = estimate_flops(model, example_input)
    finally:
        model.train(was_training)

    return ModelProfile(
        total_params=total_params,
        trainable_params=trainable_params,
        flops=flops,
        total_params_m=total_params / 1e6,
        trainable_params_m=trainable_params / 1e6,
        gflops=flops / 1e9,
    )


def log_model_profile(model: nn.Module, example_input: torch.Tensor, model_name: str = "model") -> ModelProfile:
    profile = profile_model(model=model, example_input=example_input)

    logger.info("==================================================")
    logger.info(f"Model Profile: {model_name}")
    logger.info(f"  - Total Params:              {profile.total_params_m:.3f} M")
    logger.info(f"  - Trainable Params:          {profile.trainable_params_m:.3f} M")
    logger.info(f"  - FLOPs / Clip:              {profile.gflops:.3f} G")
    logger.info("==================================================")

    return profile
