import logging
import os
import ray
import torch
import time
from typing import Dict, List, Optional, Tuple, TypedDict

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers

from nemo_rl.environments.cuda.cuda_verifier import eval_kernel_against_ref, check_metadata_serializable_all_types
from nemo_rl.environments.cuda.cuda_model_loader import build_compile_cache
from nemo_rl.environments.cuda.cuda_utils import extract_first_code, extract_last_code

class CudaEnvConfig(TypedDict):
    num_workers: int
    build_base_dir: str  # Base directory for CUDA compilation artifacts
    timeout: int  # Timeout for individual kernel verification
    compilation_timeout: Optional[int]  # Separate timeout for compilation phase
    stop_strings: Optional[List[str]]  # Default stop strings for this env
    gpu_arch: Optional[str]  # CUDA architecture (e.g., "8.0", "7.5")
    gpu_memory_fraction: Optional[float]  # GPU memory fraction per worker
    measure_performance: Optional[bool]  # Whether to benchmark correct kernels
    num_correctness_trials: Optional[int]  # Number of correctness trials per kernel
    num_performance_trials: Optional[int]  # Number of performance trials per kernel
    verbose: Optional[bool]  # Verbose compilation and execution logging
    cleanup_build_dirs: Optional[bool]  # Whether to cleanup build dirs after verification
    max_concurrent_compilations: Optional[int]  # Limit concurrent compilations
    retry_compilation_failures: Optional[bool]  # Whether to retry compilation failures
    cuda_device_ids: Optional[List[int]]  # Specific CUDA devices to use

class CudaEnvironmentMetadata(TypedDict):
    reference_implementation: str  # Original PyTorch model source code
    problem_id: int  # Unique problem identifier
    problem_name: str  # Human-readable problem name
    problem_description: Optional[str]  # Problem description
    num_correctness_trials: Optional[int]  # Override default number of trials
    measure_performance: Optional[bool]  # Override default performance measurement
    expected_speedup: Optional[float]  # Expected performance improvement
    gpu_memory_requirements: Optional[int]  # Minimum GPU memory required (MB)
    compilation_flags: Optional[List[str]]  # Custom compilation flags
    reference_baseline_time: Optional[float]  # Reference implementation timing

@ray.remote
class CudaVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.CUDA

    def __init__(self, 
                 worker_id: int,
                 gpu_id: int,
                 verbose: bool = False,
                 gpu_memory_fraction: Optional[float] = None):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.verbose = verbose
        
        logging.getLogger("cuda_verify").setLevel(
            logging.INFO if verbose else logging.WARNING
        )
        
        # Debug: Check environment variables in worker
        import os
        if self.verbose:
            print(f"Worker {worker_id} CUDA_VISIBLE_DEVICES before: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # FORCE set CUDA_VISIBLE_DEVICES to only this worker's assigned GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        if self.verbose:
            print(f"Worker {worker_id} CUDA_VISIBLE_DEVICES after: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"Worker {worker_id} torch.cuda.is_available(): {torch.cuda.is_available()}")
            print(f"Worker {worker_id} torch.cuda.device_count(): {torch.cuda.device_count()}")
        
        # Set up CUDA device (since worker only sees one GPU, use device 0)
        self.device = torch.device("cuda:0")
        
        # Verify CUDA is available before setting device
        if not torch.cuda.is_available():
            raise RuntimeError(f"Worker {worker_id}: CUDA is not available in Ray worker")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError(f"Worker {worker_id}: No CUDA devices visible (assigned GPU {gpu_id})")
        
        if self.verbose:
            print(f"Worker {worker_id} attempting to set device to cuda:0 (physical GPU {gpu_id})")
        
        # Set the device
        torch.cuda.set_device(self.device)
        
        # Test if we can actually use the device
        try:
            test_tensor = torch.tensor([1.0], device=self.device)
            test_tensor.cpu()  # This will fail if device is not accessible
            if self.verbose:
                print(f"Worker {worker_id} successfully created tensor on GPU {gpu_id}")
        except Exception as e:
            raise RuntimeError(f"Worker {worker_id}: Cannot create tensor on GPU {gpu_id}: {e}") from e
        
        # Set GPU memory fraction if specified
        if gpu_memory_fraction is not None:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
        
        # Import verification functions
        self.verify_func = eval_kernel_against_ref
        self.compile_func = build_compile_cache

    def verify(
        self,
        pred_responses: List[str],
        metadata: List[CudaEnvironmentMetadata],
        timeout: int,
        build_base_dir: str,
        compilation_timeout: Optional[int] = None,
        measure_performance: bool = False,
        num_correctness_trials: int = 1,
        num_performance_trials: int = 10,
    ) -> List[float]:
        """Verify the correctness of CUDA kernels against reference implementations.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            metadata: List[CudaEnvironmentMetadata]. The metadata containing reference implementations.
            timeout: int. Timeout for overall verification.
            build_base_dir: str. Base directory for build artifacts.
            compilation_timeout: Optional[int]. Timeout for compilation phase.
            measure_performance: bool. Whether to measure performance of correct kernels.
            num_correctness_trials: int. Number of correctness trials per kernel.
            num_performance_trials: int. Number of performance trials per kernel.

        Returns:
            List[float]. The rewards for each predicted response.
        """
        results = []
        
        for i, (response, metadata_item) in enumerate(zip(pred_responses, metadata)):
            try:
                # Extract CUDA code from response - complete KernelBench extraction logic
                final_response = response.split("</think>")[-1].strip()  # exclude <think> </think> tags
                
                # Try different extraction methods (following KernelBench patterns)
                cuda_code = extract_first_code(final_response, ["python", "cpp", "cuda"])
                if not cuda_code:
                    cuda_code = extract_last_code(final_response, ["python", "cpp", "cuda"])
                
                if not cuda_code:
                    if self.verbose:
                        print(f"Worker {self.worker_id}: No CUDA code found in response for sample {i}")
                    results.append(0.0)
                    continue

                # Create unique build directory for this worker and sample
                build_dir = os.path.join(
                    build_base_dir, 
                    f"worker_{self.worker_id}", 
                    f"sample_{i}_{int(time.time())}"
                )
                os.makedirs(build_dir, exist_ok=True)

                # Get reference implementation
                reference_src = metadata_item["reference_implementation"]
                
                # Phase 1: Pre-compilation (can be done without GPU)
                if compilation_timeout:
                    # Use subprocess-based compilation with timeout
                    compiled, stdout_content, error_msg = self.compile_func(
                        cuda_code, 
                        verbose=self.verbose, 
                        build_dir=build_dir
                    )
                else:
                    # Use direct compilation
                    compiled, stdout_content, error_msg = self.compile_func(
                        cuda_code, 
                        verbose=self.verbose, 
                        build_dir=build_dir
                    )

                if not compiled:
                    if self.verbose:
                        print(f"Worker {self.worker_id}: Compilation failed for sample {i}: {error_msg}")
                    results.append(0.0)
                    continue

                # Phase 2: GPU execution and verification
                num_trials = metadata_item.get("num_correctness_trials", num_correctness_trials)
                measure_perf = metadata_item.get("measure_performance", measure_performance)
                
                kernel_result = self.verify_func(
                    original_model_src=reference_src,
                    custom_model_src=cuda_code,
                    seed_num=42,
                    num_correct_trials=num_trials,
                    num_perf_trials=num_performance_trials,
                    verbose=self.verbose,
                    measure_performance=measure_perf,
                    build_dir=build_dir,
                    device=self.device,
                )

                # Handle retry cases (lock file errors, etc.)
                if kernel_result is None:
                    if self.verbose:
                        print(f"Worker {self.worker_id}: Verification returned None for sample {i} - retry needed")
                    results.append(0.0)  # Could implement retry logic here
                    continue

                # Convert result to reward
                if kernel_result.compiled and kernel_result.correctness:
                    results.append(1.0)
                    if self.verbose:
                        print(f"Worker {self.worker_id}: Sample {i} PASSED verification")
                else:
                    results.append(0.0)
                    if self.verbose:
                        print(f"Worker {self.worker_id}: Sample {i} FAILED - compiled: {kernel_result.compiled}, correct: {kernel_result.correctness}")

                # Clean up build directory if configured
                # (Could be made configurable via CudaEnvConfig)
                
            except Exception as e:
                if self.verbose:
                    print(f"Worker {self.worker_id}: Error verifying sample {i}: {e}")
                results.append(0.0)

        return results

    def get_worker_stats(self) -> Dict:
        """Get worker statistics and GPU memory usage."""
        with torch.cuda.device(self.device):
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "device": str(self.device),
            "memory_allocated_mb": memory_allocated / (1024 * 1024),
            "memory_reserved_mb": memory_reserved / (1024 * 1024),
        }
    

@ray.remote
class CudaEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.CUDA

    def __init__(self, cfg: CudaEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.build_base_dir = cfg["build_base_dir"]
        
        # Set up GPU architecture if specified
        if cfg.get("gpu_arch"):
            from nemo_rl.environments.cuda.cuda_utils import set_gpu_arch
            set_gpu_arch(cfg["gpu_arch"])
        
        # Determine GPU assignments
        if cfg.get("cuda_device_ids"):
            available_gpus = cfg["cuda_device_ids"]
        else:
            # Try to detect available CUDA devices
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    available_gpus = list(range(device_count))
                else:
                    # Fallback: assume at least device 0 if CUDA is available
                    available_gpus = [0]
            else:
                available_gpus = []
        
        if not available_gpus:
            raise RuntimeError("No CUDA devices available for CudaEnvironment")
        
        # Log GPU assignment info if verbose
        if cfg.get("verbose", False):
            print(f"CudaEnvironment: Creating {self.num_workers} workers across {len(available_gpus)} GPUs")
            print(f"Available GPU IDs: {available_gpus}")
        
        # Create workers with GPU assignment (round-robin)
        self.workers = []
        for i in range(self.num_workers):
            gpu_id = available_gpus[i % len(available_gpus)]
            
            if cfg.get("verbose", False):
                print(f"Worker {i} -> GPU {gpu_id}")
            
            worker = CudaVerifyWorker.options(
                num_gpus=1,  # Each worker requests 1 GPU from Ray
                runtime_env={
                    "py_executable": CudaVerifyWorker.DEFAULT_PY_EXECUTABLE,
                    "env_vars": {
                        "TORCH_USE_CUDA_DSA": "1",  # Enable device-side assertions
                    }
                }
            ).remote(
                worker_id=i,
                gpu_id=gpu_id,
                verbose=cfg.get("verbose", False),
                gpu_memory_fraction=cfg.get("gpu_memory_fraction")
            )
            self.workers.append(worker)

        # Create base build directory
        os.makedirs(self.build_base_dir, exist_ok=True)

    def shutdown(self):
        """Shutdown all CUDA workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[CudaEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the CUDA environment.

        Args:
            message_log_batch: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs.
            metadata: List[CudaEnvironmentMetadata]. CUDA-specific metadata with reference implementations.

        Returns:
            EnvironmentReturn: Contains observations, metadata, stop strings, rewards, and done flags.
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        # Create unique build directory for this batch
        batch_build_dir = os.path.join(
            self.build_base_dir, 
            f"batch_{int(time.time() * 1000)}"  # Use milliseconds for uniqueness
        )

        # Chunk work across workers
        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_metadata = chunk_list_to_workers(metadata, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(
                pred_responses=chunk, 
                metadata=metadata_chunk, 
                timeout=self.cfg["timeout"],
                build_base_dir=batch_build_dir,
                compilation_timeout=self.cfg.get("compilation_timeout"),
                measure_performance=self.cfg.get("measure_performance", False),
                num_correctness_trials=self.cfg.get("num_correctness_trials", 1),
                num_performance_trials=self.cfg.get("num_performance_trials", 10),
            )
            for i, (chunk, metadata_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_metadata)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
        observations = [
            {
                "role": "environment",
                "content": "CUDA kernel compiled and correct"
                if result
                else "CUDA kernel failed compilation or correctness",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this CUDA environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
            # CUDA-specific metrics
            "cuda_compilation_success_rate": self._calculate_compilation_success_rate(batch),
            "cuda_kernel_correctness_rate": batch["rewards"].mean().item(),
        }

        return batch, metrics

    def _calculate_compilation_success_rate(self, batch: BatchedDataDict) -> float:
        """Calculate CUDA compilation success rate from batch data."""
        # This would need to be implemented based on how you track compilation vs correctness failures
        # For now, returning the same as accuracy since we don't separate compilation from correctness in rewards
        return batch["rewards"].mean().item()