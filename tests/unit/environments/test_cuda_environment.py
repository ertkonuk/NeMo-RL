# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
import time

import pytest
import ray
import torch

from nemo_rl.environments.cuda_environment import CudaEnvironment


@pytest.fixture(scope="module")
def cuda_env():
    """Create a CudaEnvironment actor for testing."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")
    
    # Check if we have actual CUDA devices
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available for testing")
    
    # Test if we can actually use CUDA in Ray environment first
    @ray.remote
    def test_cuda_in_ray(device_id):
        import os
        try:
            import torch
            # Set CUDA_VISIBLE_DEVICES to make GPUs visible
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
            if not torch.cuda.is_available():
                return False, "CUDA not available in Ray worker"
            if torch.cuda.device_count() <= device_id:
                return False, f"Device {device_id} not available in Ray worker"
            torch.cuda.set_device(device_id)
            test_tensor = torch.tensor([1.0], device=f"cuda:{device_id}")
            test_tensor.cpu()
            return True, "CUDA accessible in Ray worker"
        except Exception as e:
            return False, str(e)
    
    # Test CUDA accessibility in Ray worker with proper environment
    cuda_test_result = ray.get(test_cuda_in_ray.options(runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}).remote(0))
    
    if not cuda_test_result[0]:
        pytest.skip(f"CUDA not accessible in Ray workers: {cuda_test_result[1]}")
    
    # Create temporary build directory
    build_dir = tempfile.mkdtemp(prefix="cuda_test_")
    
    try:
        # Ensure CUDA is visible to Ray workers - copy host environment and set all GPUs
        env_vars = dict(os.environ)
        env_vars["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Environment level - workers will override per-GPU
        
        env = CudaEnvironment.options(
            runtime_env={
                "py_executable": CudaEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": env_vars,
            }
        ).remote({
            "num_workers": 8,  # 8 workers for 8 GPUs (1 GPU per worker)
            "build_base_dir": build_dir,
            "timeout": 30,
            "compilation_timeout": 20,
            "verbose": True,
            "measure_performance": False,
            "num_correctness_trials": 1,
            "num_performance_trials": 3,
            "cuda_device_ids": [0, 1, 2, 3, 4, 5, 6, 7],  # All 8 GPUs
        })
        yield env
        # Clean up the actor and wait for it to be killed
        env.shutdown.remote()
        ray.kill(env)
        # Give some time for cleanup
        time.sleep(0.1)
    except Exception as e:
        pytest.skip(f"Failed to create CudaEnvironment: {e}")
    
    # Clean up build directory
    import shutil
    try:
        shutil.rmtree(build_dir)
    except:
        pass  # Best effort cleanup


@pytest.fixture
def hinge_loss_test_data():
    """Test data for HingeLoss CUDA kernel optimization."""
    # Reference implementation from KernelBench
    reference_implementation = '''import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))

batch_size = 128
input_shape = (1,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]

def get_init_inputs():
    return []'''

    # Correct CUDA implementation that should pass verification
    correct_cuda_implementation = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for hinge loss
hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, 
                                  float* losses, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float loss = fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
        losses[idx] = loss;
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // Create output tensor with same shape as predictions
    torch::Tensor losses = torch::empty_like(predictions);
    const int block_size = 256;
    const int num_blocks = (predictions.numel() + block_size - 1) / block_size;
    
    hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        losses.data_ptr<float>(), 
        predictions.numel()
    );
    
    return losses.mean();
}
"""

hinge_loss_cpp_source = "torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code for hinge loss
hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=hinge_loss_cpp_source,
    cuda_sources=hinge_loss_source,
    functions=["hinge_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.hinge_loss = hinge_loss
    
    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)'''

    # Incorrect implementation (wrong formula) that should fail verification
    incorrect_cuda_implementation = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define incorrect CUDA kernel (missing the max operation)
wrong_hinge_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void wrong_hinge_loss_kernel(const float* predictions, const float* targets, 
                                        float* losses, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Wrong: missing max(0, ...) operation
        losses[idx] = 1.0f - predictions[idx] * targets[idx];
    }
}

torch::Tensor wrong_hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    torch::Tensor losses = torch::empty_like(predictions);
    const int block_size = 256;
    const int num_blocks = (predictions.numel() + block_size - 1) / block_size;
    
    wrong_hinge_loss_kernel<<<num_blocks, block_size>>>(
        predictions.data_ptr<float>(), 
        targets.data_ptr<float>(), 
        losses.data_ptr<float>(), 
        predictions.numel()
    );
    
    return losses.mean();
}
"""

wrong_hinge_loss_cpp_source = "torch::Tensor wrong_hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

# Compile the inline CUDA code
wrong_hinge_loss = load_inline(
    name="wrong_hinge_loss",
    cpp_sources=wrong_hinge_loss_cpp_source,
    cuda_sources=wrong_hinge_loss_source,
    functions=["wrong_hinge_loss_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.wrong_hinge_loss = wrong_hinge_loss
    
    def forward(self, predictions, targets):
        return self.wrong_hinge_loss.wrong_hinge_loss_cuda(predictions, targets)'''

    # Compilation failure case (syntax error)
    compilation_failure_implementation = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel with syntax error
bad_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void bad_kernel(const float* predictions, const float* targets, 
                           float* losses, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Syntax error: missing semicolon
        losses[idx] = fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx])
    }
}

torch::Tensor bad_cuda(torch::Tensor predictions, torch::Tensor targets) {
    // This will fail to compile
    return torch::empty_like(predictions);
}
"""

bad_cpp_source = "torch::Tensor bad_cuda(torch::Tensor predictions, torch::Tensor targets);"

bad_kernel = load_inline(
    name="bad_kernel",
    cpp_sources=bad_cpp_source,
    cuda_sources=bad_cuda_source,
    functions=["bad_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.bad_kernel = bad_kernel
    
    def forward(self, predictions, targets):
        return self.bad_kernel.bad_cuda(predictions, targets)'''

    return {
        "correct_case": {
            "message_log_batch": [
                [
                    {
                        "role": "user",
                        "content": "Optimize this HingeLoss model with custom CUDA kernels:"
                    },
                    {
                        "role": "assistant", 
                        "content": f"```python\n{correct_cuda_implementation}\n```"
                    }
                ]
            ],
            "metadata": [
                {
                    "reference_implementation": reference_implementation,
                    "problem_id": 100,
                    "problem_name": "HingeLoss",
                    "num_correctness_trials": 1,
                    "measure_performance": False,
                }
            ]
        },
        "incorrect_case": {
            "message_log_batch": [
                [
                    {
                        "role": "user",
                        "content": "Optimize this HingeLoss model with custom CUDA kernels:"
                    },
                    {
                        "role": "assistant",
                        "content": f"```python\n{incorrect_cuda_implementation}\n```"
                    }
                ]
            ],
            "metadata": [
                {
                    "reference_implementation": reference_implementation,
                    "problem_id": 100,
                    "problem_name": "HingeLoss",
                    "num_correctness_trials": 1,
                    "measure_performance": False,
                }
            ]
        },
        "compilation_failure_case": {
            "message_log_batch": [
                [
                    {
                        "role": "user",
                        "content": "Optimize this HingeLoss model with custom CUDA kernels:"
                    },
                    {
                        "role": "assistant",
                        "content": f"```python\n{compilation_failure_implementation}\n```"
                    }
                ]
            ],
            "metadata": [
                {
                    "reference_implementation": reference_implementation,
                    "problem_id": 100,
                    "problem_name": "HingeLoss",
                    "num_correctness_trials": 1,
                    "measure_performance": False,
                }
            ]
        }
    }


@pytest.fixture
def mixed_cuda_test_data(hinge_loss_test_data):
    """Test data with mix of correct, incorrect, and compilation failure cases."""
    return {
        "message_log_batch": [
            hinge_loss_test_data["correct_case"]["message_log_batch"][0],
            hinge_loss_test_data["incorrect_case"]["message_log_batch"][0], 
            hinge_loss_test_data["compilation_failure_case"]["message_log_batch"][0],
        ],
        "metadata": [
            hinge_loss_test_data["correct_case"]["metadata"][0],
            hinge_loss_test_data["incorrect_case"]["metadata"][0],
            hinge_loss_test_data["compilation_failure_case"]["metadata"][0],
        ]
    }


def test_cuda_env_step_correct_implementation(cuda_env, hinge_loss_test_data):
    """Test CudaEnvironment step with correct CUDA implementation."""
    test_data = hinge_loss_test_data["correct_case"]
    
    result = ray.get(
        cuda_env.step.remote(
            test_data["message_log_batch"], 
            test_data["metadata"]
        )
    )

    # Check observations
    assert len(result.observations) == 1, "Should return observation for 1 message"
    assert result.observations[0]["role"] == "environment", "Observation should be from environment"
    assert "compiled and correct" in result.observations[0]["content"], (
        "Should indicate successful compilation and correctness"
    )

    # Check metadata (should be unchanged)
    assert len(result.metadata) == 1, "Should return metadata for 1 message"
    assert result.metadata == test_data["metadata"], "Metadata should be unchanged"

    # Check rewards and termination flags
    assert result.rewards.shape == (1,), "Rewards should be tensor of shape (1,)"
    assert result.rewards[0] == 1.0, "Reward should be 1.0 for correct implementation"
    assert result.terminateds.shape == (1,), "Terminated flags should be tensor of shape (1,)"
    assert result.terminateds[0] == 1.0, "Terminated flag should be 1.0"


def test_cuda_env_step_incorrect_implementation(cuda_env, hinge_loss_test_data):
    """Test CudaEnvironment step with incorrect CUDA implementation."""
    test_data = hinge_loss_test_data["incorrect_case"]
    
    result = ray.get(
        cuda_env.step.remote(
            test_data["message_log_batch"],
            test_data["metadata"]
        )
    )

    # Check observations
    assert len(result.observations) == 1, "Should return observation for 1 message"
    assert result.observations[0]["role"] == "environment", "Observation should be from environment"
    assert "failed" in result.observations[0]["content"], (
        "Should indicate failure in compilation or correctness"
    )

    # Check rewards
    assert result.rewards.shape == (1,), "Rewards should be tensor of shape (1,)"
    assert result.rewards[0] == 0.0, "Reward should be 0.0 for incorrect implementation"


def test_cuda_env_step_compilation_failure(cuda_env, hinge_loss_test_data):
    """Test CudaEnvironment step with compilation failure."""
    test_data = hinge_loss_test_data["compilation_failure_case"]
    
    result = ray.get(
        cuda_env.step.remote(
            test_data["message_log_batch"],
            test_data["metadata"]
        )
    )

    # Check observations
    assert len(result.observations) == 1, "Should return observation for 1 message"
    assert result.observations[0]["role"] == "environment", "Observation should be from environment"
    assert "failed" in result.observations[0]["content"], (
        "Should indicate compilation failure"
    )

    # Check rewards
    assert result.rewards.shape == (1,), "Rewards should be tensor of shape (1,)"
    assert result.rewards[0] == 0.0, "Reward should be 0.0 for compilation failure"


def test_cuda_env_step_mixed(cuda_env, mixed_cuda_test_data):
    """Test CudaEnvironment step with mix of correct, incorrect, and compilation failure."""
    result = ray.get(
        cuda_env.step.remote(
            mixed_cuda_test_data["message_log_batch"],
            mixed_cuda_test_data["metadata"]
        )
    )

    # Check observations
    assert len(result.observations) == 3, "Should return observations for all 3 messages"
    assert all(obs["role"] == "environment" for obs in result.observations), (
        "All observations should be from environment"
    )

    # Check rewards
    assert result.rewards.shape == (3,), "Rewards should be tensor of shape (3,)"
    assert result.rewards[0] == 1.0, "First (correct) implementation should get reward 1.0"
    assert result.rewards[1] == 0.0, "Second (incorrect) implementation should get reward 0.0"
    assert result.rewards[2] == 0.0, "Third (compilation failure) should get reward 0.0"

    # Check metadata
    assert len(result.metadata) == 3, "Should return metadata for all 3 messages"
    assert result.metadata == mixed_cuda_test_data["metadata"], "Metadata should be unchanged"


def test_cuda_env_step_empty(cuda_env):
    """Test CudaEnvironment step with empty input."""
    result = ray.get(cuda_env.step.remote([], []))

    # Check all outputs are empty
    assert len(result.observations) == 0, "Should return empty observations list"
    assert len(result.metadata) == 0, "Should return empty metadata list"
    assert result.rewards.shape == (0,), "Should return empty rewards tensor"
    assert result.terminateds.shape == (0,), "Should return empty terminateds tensor"


def test_cuda_env_no_code_extraction(cuda_env, hinge_loss_test_data):
    """Test CudaEnvironment with response that contains no extractable CUDA code."""
    test_data = hinge_loss_test_data["correct_case"]
    
    # Create test case with no code blocks
    no_code_data = {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Optimize this HingeLoss model with custom CUDA kernels:"
                },
                {
                    "role": "assistant",
                    "content": "I need more time to think about this optimization."
                }
            ]
        ],
        "metadata": test_data["metadata"]
    }
    
    result = ray.get(
        cuda_env.step.remote(
            no_code_data["message_log_batch"],
            no_code_data["metadata"]
        )
    )

    # Should fail due to no extractable code
    assert result.rewards[0] == 0.0, "Should get 0.0 reward when no code is extracted"
    assert "failed" in result.observations[0]["content"], "Should indicate failure"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_env_multiple_assistant_messages(cuda_env, hinge_loss_test_data):
    """Test CudaEnvironment with multiple assistant messages (should use combined response)."""
    test_data = hinge_loss_test_data["correct_case"]
    
    # Split the CUDA code across multiple assistant messages
    cuda_code = test_data["message_log_batch"][0][1]["content"]
    
    multi_message_data = {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Optimize this HingeLoss model with custom CUDA kernels:"
                },
                {
                    "role": "assistant",
                    "content": "Let me think about this step by step..."
                },
                {
                    "role": "assistant",
                    "content": cuda_code
                }
            ]
        ],
        "metadata": test_data["metadata"]
    }
    
    result = ray.get(
        cuda_env.step.remote(
            multi_message_data["message_log_batch"],
            multi_message_data["metadata"]
        )
    )

    # Should still work correctly by combining assistant messages
    assert len(result.observations) == 1, "Should return observation for 1 conversation"
    # Note: This might pass or fail depending on the exact implementation details
    # The key is that it should handle multiple assistant messages gracefully 