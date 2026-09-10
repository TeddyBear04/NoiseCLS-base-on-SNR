import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import logging
import torch
import torch.nn as nn

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceTimer:
    """
    OOP compliant InferenceTimer to measure model inference latency and throughput.
    Performs device warm-up (CUDA kernel caching) before final measurements.
    """
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        """
        Initialize InferenceTimer.

        Args:
            model (nn.Module): The model instance to measure.
            device (torch.device): Hardware device running the model (CPU or CUDA GPU).
        """
        self.model = model
        self.device = device

    def measure_latency_per_sample(
        self,
        sample_length: int = 128000,
        warm_up_steps: int = 10,
        num_steps: int = 50
    ) -> float:
        """
        Measure average inference latency per single sample.

        Args:
            sample_length (int): Length of raw input audio waveform (Default: 128000 - equivalent to 2s at 64kHz).
            warm_up_steps (int): Number of warm-up dummy runs to cache CUDA. Default: 10.
            num_steps (int): Number of iterations for average measurement. Default: 50.

        Returns:
            float: Average inference latency per sample in milliseconds (ms).
        """
        # Create a single dummy waveform sample [1, sample_length]
        dummy_input = torch.randn(1, sample_length).to(self.device)
        self.model.eval()
        
        # Check if CUDA device is active
        is_cuda = self.device.type == 'cuda'

        # 1. Warm-up Phase
        logger.info(f"Starting Warm-up phase with {warm_up_steps} steps...")
        with torch.no_grad():
            for _ in range(warm_up_steps):
                _ = self.model(dummy_input)
                if is_cuda:
                    torch.cuda.synchronize()
        logger.info("Warm-up phase completed.")

        # 2. Measurement Phase
        logger.info(f"Measuring average Inference Latency over {num_steps} iterations...")
        
        if is_cuda:
            torch.cuda.synchronize()
            
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_steps):
                _ = self.model(dummy_input)
                if is_cuda:
                    torch.cuda.synchronize()
                    
        end_time = time.perf_counter()
        
        # Calculate latency and throughput metrics
        total_time_seconds = end_time - start_time
        latency_per_sample_ms = (total_time_seconds / num_steps) * 1000.0
        throughput_fps = 1000.0 / latency_per_sample_ms
        
        logger.info("==================================================")
        logger.info("INFERENCE PERFORMANCE REPORT (LATENCY REPORT):")
        logger.info(f"  - Device:                         {self.device}")
        logger.info(f"  - Avg Inference Latency / sample: {latency_per_sample_ms:.3f} ms")
        logger.info(f"  - Throughput:                     {throughput_fps:.1f} samples/sec")
        logger.info("==================================================")
        
        return latency_per_sample_ms
