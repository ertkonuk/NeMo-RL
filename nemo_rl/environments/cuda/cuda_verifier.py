"""
Complete extraction of KernelBench verification functionality.
"""
import torch
import torch.nn as nn
import os
import json
import statistics
from typing import Dict, List, Optional, NamedTuple
from pydantic import BaseModel

from nemo_rl.environments.cuda.cuda_model_loader import (
    load_original_model_and_inputs, load_custom_model, graceful_eval_cleanup, set_seed
)

class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    COMPLETE extraction from KernelBench eval.py - no changes
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    COMPLETE extraction from KernelBench eval.py - no changes
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata

def time_execution_with_cuda_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a kernel function using CUDA events for precise GPU timing.
    COMPLETE extraction from KernelBench eval.py - no changes
    
    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to the function
        num_warmup: Number of warmup runs
        num_trials: Number of timing trials
        verbose: Whether to print timing info
        device: CUDA device to use
        
    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        device = torch.cuda.current_device()
    
    torch.cuda.set_device(device)
    
    # Warmup runs
    for _ in range(num_warmup):
        with torch.cuda.device(device):
            _ = kernel_fn(*args)
            torch.cuda.synchronize(device=device)
    
    # Actual timing runs
    elapsed_times = []
    
    for trial in range(num_trials):
        # Create CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.cuda.device(device):
            start_event.record()
            _ = kernel_fn(*args)
            end_event.record()
            
            torch.cuda.synchronize(device=device)
            
            # Get elapsed time in milliseconds
            elapsed_time = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time)
            
            if verbose:
                print(f"Trial {trial + 1}: {elapsed_time:.4f} ms")
    
    return elapsed_times

def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    COMPLETE extraction from KernelBench eval.py - no changes
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.cuda(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.cuda(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    COMPLETE extraction from KernelBench eval.py - no changes
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata

def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    COMPLETE extraction from KernelBench eval.py - no changes
    """
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata

def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.
    COMPLETE extraction from KernelBench eval.py - no changes

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """
    if not elapsed_times:
        return {
            "mean": -1.0,
            "std": -1.0,
            "min": -1.0,
            "max": -1.0,
            "num_trials": 0,
            "device": str(device) if device else "unknown"
        }
    
    stats = {
        "mean": statistics.mean(elapsed_times),
        "std": statistics.stdev(elapsed_times) if len(elapsed_times) > 1 else 0.0,
        "min": min(elapsed_times),
        "max": max(elapsed_times),
        "num_trials": len(elapsed_times),
        "device": str(device) if device else "unknown",
        "all_times": elapsed_times  # Keep all individual measurements
    }
    
    return stats

def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: torch.device = torch.cuda.current_device() if torch.cuda.is_available() else None, # have to run on GPU
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    COMPLETE extraction from KernelBench eval.py - no changes
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # add hash for later to distinguish between multi-turn kernels
        ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device)
            return None
        else:
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context, device)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device)
        metadata["runtime_error"] = e
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                model_new = custom_model.cuda(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_cuda_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    graceful_eval_cleanup(context, device)
    return kernel_exec_result