import logging
import os
import ray
import shutil
import time
import hashlib
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

# Ray scheduling helpers
from ray.util.scheduling_strategies import (
    PlacementGroupSchedulingStrategy,
)

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

# Torch is needed for CPU-side tensor ops in the env actor; GPU workers re-import it after Ray sets CVD
import torch

class CudaEnvConfig(TypedDict):
    # Core Ray/Cluster settings
    num_workers: int
    cuda_device_ids: Optional[List[int]]  # Specific CUDA devices to use
    gpu_memory_fraction: Optional[float]  # GPU memory fraction per worker

    # Build/Storage related settings
    cuda_build_cache: Optional[str] = "cuda_build_cache"  # Base directory for CUDA compilation artifacts

    # Verification settings
    timeout: int  # Timeout for individual kernel verification
    compilation_timeout: Optional[int]  # Separate timeout for compilation phase
    gpu_arch: Optional[str]  =  "Hopper" # CUDA architecture (e.g., "Hopper", "Ampere", "Ada")
    measure_performance: Optional[bool]  # Whether to benchmark correct kernels
    num_correctness_trials: Optional[int]  # Number of correctness trials per kernel
    num_performance_trials: Optional[int]  # Number of performance trials per kernel
    max_concurrent_compilations: Optional[int]  # Limit concurrent compilations

    # Misc environmentsettings
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env
    verbose: Optional[bool] = True # Verbose compilation and execution logging


class CudaEnvironmentMetadata(TypedDict):
    reference_implementation: str  # Original PyTorch model source code

def prepare_cuda_build_info(metadata, conversation_content, sample_index):
    """Preprocess metadata to add unique build directory info."""
    # Create unique build identifier
    content_hash = hashlib.md5(str(conversation_content).encode()).hexdigest()[:8] 
    timestamp = int(time.time() * 1000000)  # microseconds
    unique_build_id = f"sample_{sample_index}_{content_hash}_{timestamp}"
    
    # Enrich metadata with build info
    build_metadata = dict(metadata)  # Copy original
    build_metadata["build_unique_id"] = unique_build_id
    build_metadata["sample_index"] = sample_index
    
    return build_metadata

@ray.remote
class CudaVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.CUDA

    def __init__(self,
                 worker_id: int,
                 cuda_build_cache: str,
                 verbose: bool = True,
                 gpu_memory_fraction: Optional[float] = None):
        self.worker_id = worker_id
        self.verbose = verbose
        
        logging.getLogger("cuda_verify").setLevel(
            logging.INFO if verbose else logging.WARNING
        )

        # Create worker's unique base directory
        worker_timestamp = int(time.time() * 1000000)  # Use already imported time
        self.base_dir = os.path.join(
            cuda_build_cache,
            f"worker_{worker_id}_{worker_timestamp}"
        )
        os.makedirs(self.base_dir, exist_ok=True)

        # Debug: Check environment variables in worker
        if self.verbose:
            print(f"Worker {worker_id} CUDA_VISIBLE_DEVICES before: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # Import torch *after* Ray has set CUDA_VISIBLE_DEVICES automatically.
        import torch  # pylint: disable=import-error,import-outside-toplevel

        # Ray exposes exactly one GPU, so cuda:0 is always correct.
        self.device = torch.device("cuda:0")

        # Sanity-check CUDA availability (should always be True now)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Worker {worker_id}: CUDA unavailable despite Ray allocating a GPU."
            )

        if self.verbose:
            print(
                f"Worker {worker_id}: torch.cuda.device_count() = {torch.cuda.device_count()}"
            )
        
        # Test device usability
        try:
            _ = torch.tensor([1.0], device=self.device).cpu()
        except Exception as e:
            raise RuntimeError(
                f"Worker {worker_id}: Failed to allocate tensor on CUDA device: {e}"
            ) from e
        
        # Set GPU memory fraction if specified
        if gpu_memory_fraction is not None and hasattr(torch.cuda, "set_per_process_memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
        
        # Import verification functions
        self.verify_func = eval_kernel_against_ref
        self.compile_func = build_compile_cache
    
    def _cleanup_build_dir(self, build_dir: str) -> None:
        """Clean up build directory after verification (KernelBench style)."""
        try:
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
                if self.verbose:
                    print(f"Worker {self.worker_id}: Cleaned up build directory: {build_dir}")
        except Exception as e:
            if self.verbose:
                print(f"Worker {self.worker_id}: Warning - Failed to cleanup {build_dir}: {e}")
            # Don't raise - cleanup failure shouldn't break training

    def verify(
        self,
        pred_responses: List[str],
        metadata: List[CudaEnvironmentMetadata],
        timeout: int,
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

                # Create unique build directory using the preprocessed build ID
                build_unique_id = metadata_item.get("build_unique_id", f"sample_{i}_{int(time.time())}")
                build_dir = os.path.join(self.base_dir, build_unique_id)
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
                
                kernel_result = self.verify_func(
                    original_model_src=reference_src,
                    custom_model_src=cuda_code,
                    seed_num=42,
                    num_correct_trials=num_correctness_trials,
                    num_perf_trials=num_performance_trials,
                    verbose=self.verbose,
                    measure_performance=measure_performance,
                    build_dir=build_dir,
                    device=self.device,
                )

                # Handle retry cases (lock file errors, etc.)
                if kernel_result is None:
                    if self.verbose:
                        print(f"Worker {self.worker_id}: Verification returned None for sample {i} - retry needed")
                    results.append(0.0)  # Could implement retry logic here
                    continue

                # --------------------------------------------------------------
                # Convert verification result into a scalar reward
                # --------------------------------------------------------------
                if kernel_result.compiled and kernel_result.correctness:
                    reward_val = 1.0  # default for correctness only

                    if measure_performance:
                        # Use speed-up over baseline as reward when available
                        speedup = kernel_result.metadata.get("speedup")
                        if speedup is not None and isinstance(speedup, (int, float)) and speedup > 0:
                            reward_val = float(speedup)

                    results.append(reward_val)
                    if self.verbose:
                        print(
                            f"Worker {self.worker_id}: Sample {i} PASSED verification | Reward = {reward_val:.4f}"
                        )
                else:
                    results.append(0.0)
                    if self.verbose:
                        print(f"Worker {self.worker_id}: Sample {i} FAILED - compiled: {kernel_result.compiled}, correct: {kernel_result.correctness}")

                # Always clean up build directory after verification
                self._cleanup_build_dir(build_dir)
                
            except Exception as e:
                if self.verbose:
                    print(f"Worker {self.worker_id}: Error verifying sample {i}: {e}")
                results.append(0.0)
                # Clean up build directory even on exception
                if 'build_dir' in locals():
                    self._cleanup_build_dir(build_dir)

        return results

    def get_worker_stats(self) -> Dict:
        """Get worker statistics and GPU memory usage."""
        import torch  # pylint: disable=import-error,import-outside-toplevel

        with torch.cuda.device(self.device):
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            
        return {
            "worker_id": self.worker_id,
            "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
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
        self.cuda_build_cache = cfg["cuda_build_cache"]
        
        # Set up GPU architecture (default to Hopper)
        from nemo_rl.environments.cuda.cuda_utils import set_gpu_arch
        set_gpu_arch(cfg.get("gpu_arch", "Hopper"))
        
        # ---- Create placement group for single-GPU verify workers (following llm_judge pattern) ----
        # Since each worker needs exactly 1 GPU (like tensor_parallel_size=1), we use placement group
        bundle_ct_per_node_list = [1] * self.num_workers  # One 1-GPU bundle per worker
        
        self.verify_vc = RayVirtualCluster(
            bundle_ct_per_node_list=bundle_ct_per_node_list,
            use_gpus=True,
            name="cuda_verify_vc",
        )
        if cfg.get("verbose", False):
            self.verify_vc.print_cluster_grid()
        placement_groups = self.verify_vc.get_placement_groups()

        # Set up worker scheduling strategy
        if self.verify_vc is not None:
            placement_group = self.verify_vc.get_placement_groups()[0]
            scheduling_kwargs = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(
                    placement_group=placement_group
                ),
            }
        else:
            # No placement group - let Ray handle scheduling
            scheduling_kwargs = {}

        worker_runtime_env = {
            "py_executable": CudaVerifyWorker.DEFAULT_PY_EXECUTABLE,
            "env_vars": {
                "TORCH_USE_CUDA_DSA": "1",
            },
        }

        self.workers = []
        for i in range(self.num_workers):
            # Single-GPU workers always use placement group (like llm_judge tensor_parallel_size=1)
            pg_index = i % len(placement_groups)
            pg = placement_groups[pg_index]
            scheduling_kwargs = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=pg)
            }
            
            worker = CudaVerifyWorker.options(
                num_gpus=1,
                runtime_env=worker_runtime_env,
                **scheduling_kwargs,
            ).remote(
                worker_id=i,
                cuda_build_cache=self.cuda_build_cache,
                verbose=cfg.get("verbose", False),
                gpu_memory_fraction=cfg.get("gpu_memory_fraction"),
            )
            self.workers.append(worker)

        # Create base build directory
        os.makedirs(self.cuda_build_cache, exist_ok=True)

    def shutdown(self):
        """Shutdown all CUDA workers."""
        for worker in self.workers:
            ray.kill(worker)
        # Release placement-group resources
        if hasattr(self, "verify_vc") and self.verify_vc is not None:
            self.verify_vc.shutdown()

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

        # No need for batch build directory since each worker has its own base directory

        # Chunk work across workers
        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )

        # Preprocess metadata just like code_environment does
        build_metadata = [
            prepare_cuda_build_info(m, conversation, i) 
            for i, (m, conversation) in enumerate(zip(metadata, message_log_batch))
        ]
        chunked_metadata = chunk_list_to_workers(build_metadata, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(
                pred_responses=chunk, 
                metadata=metadata_chunk, 
                timeout=self.cfg["timeout"],
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
        import torch  # pylint: disable=import-error,import-outside-toplevel
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