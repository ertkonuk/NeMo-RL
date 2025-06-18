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
import logging
from typing import Dict, List, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.code.livecodebench import compute_score, prepare_tests
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers, extract_code
from nemo_rl.data.interfaces import LLMMessageLogType


class CodeEnvConfig(TypedDict):
    num_workers: int
    role: str = "environment"  # Can be "user" or "environment" 
    stop_strings: Optional[List[str]] = None  # Default stop strings for this env
    timeout: int = 10  # Timeout for the code execution
    max_turns: int = 3  # Maximum turns allowed per episode
    turn_penalty: float = 0.8  # Multiplier for each additional turn (e.g., 0.8 means 20% penalty per turn)


class CodeEnvironmentMetadata(TypedDict):
    unittests: Optional[List[Dict[str, str]]]
    fn_name: Optional[str]
    current_turn: int  # Track current turn number


@ray.remote
class CodeVerifyWorker:
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, verbose: bool = False):
        logging.getLogger("code_verify").setLevel(
            logging.INFO if verbose else logging.WARNING
        )
        self.verify_func = compute_score

    def verify(
        self,
        pred_responses: List[str],
        metadata: List[CodeEnvironmentMetadata],
        timeout: int,
    ) -> List[Tuple[float, Dict]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: List[str]. The predicted responses from the LLM.
            metadata: List[CodeEnvironmentMetadata]. The metadata containing unit tests.
            timeout: int. Timeout for code execution.

        Returns:
            List[Tuple[float, Dict]]. The rewards and execution metadata for each predicted response.
        """
        results = []
        for response, metadata_item in zip(pred_responses, metadata):
            try:
                # No more output muting - let errors and debug info show!
                final_response = response.split("</think>")[
                    -1
                ].strip()  # exclude <think> </think> tags
                code_str = extract_code(final_response)

                ret_score, execution_metadata = self.verify_func(
                    code_str, metadata_item, timeout
                )

                # Store the execution metadata along with the score
                results.append((float(ret_score), execution_metadata or {}))

            except Exception as e:
                # Capture exception information for debugging
                error_metadata = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": str(e),
                }
                results.append((0.0, error_metadata))

        return results


@ray.remote
class CodeEnvironment(EnvironmentInterface):
    DEFAULT_PY_EXECUTABLE = PY_EXECUTABLES.SYSTEM

    def __init__(self, cfg: CodeEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            CodeVerifyWorker.options(
                runtime_env={
                    "py_executable": CodeVerifyWorker.DEFAULT_PY_EXECUTABLE,
                    "env_vars": {"OMP_NUM_THREADS": "1"},
                }
            ).remote()
            for _ in range(self.num_workers)
        ]
        self.runner = CodeRunner(self.workers, cfg)

    def shutdown(self):
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: List[List[Dict[str, str]]],
        metadata: List[CodeEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the code environment.

        Args:
            message_log: List[List[Dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: List[CodeEnvironmentMetadata]. The grader will use the 'unittests' and 'fn_name' keys to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - List[Dict[str, str]]: Observations/responses batch
                - List[Dict]: Updated metadata
                - List[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Use batch processing instead of individual processing
        observations, rewards, terminateds, all_stop_strings, all_next_metadata = self.runner.process_turn(
            message_log_batch, metadata
        )

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminateds, dtype=torch.bool)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_next_metadata,
            next_stop_strings=all_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminated_tensor,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, dict]:
        """Computes metrics for this environment given a global rollout batch.

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

        # Calculate average turns per prompt from metadata
        turns_per_prompt = []
        for extra_env_info in batch["extra_env_info"]:
            if extra_env_info and "current_turn" in extra_env_info:
                turns_per_prompt.append(extra_env_info["current_turn"])
            else:
                turns_per_prompt.append(1)  # Default to 1 turn if no data
        
        avg_turns_per_prompt = sum(turns_per_prompt) / len(turns_per_prompt) if turns_per_prompt else 1.0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
            "average_turns_per_prompt": avg_turns_per_prompt,
        }

        return batch, metrics


class CodeRunner:
    """Handles the turn-by-turn logic for the code environment."""
    
    def __init__(self, workers: List, cfg: CodeEnvConfig):
        self.workers = workers
        self.timeout = cfg["timeout"]
        self.max_turns = cfg["max_turns"]
        self.turn_penalty = cfg.get("turn_penalty", 0.8)  # Default to 0.8 if not specified
        self.role = cfg.get("role", "user")  # Default to "user" if not specified
        self.num_workers = len(workers)

    def _format_error_feedback(self, execution_metadata: Dict, current_turn: int, score: float, use_env_tags: bool = False) -> str:
        """Format detailed error feedback for the model."""
        # Only treat as correct if score > 0 AND no error metadata
        if score > 0 and (not execution_metadata or execution_metadata == {}):
            return "The solution is correct!"
        
        feedback_parts = [f"Your solution is incorrect."]
        
        # Handle compilation errors
        if "error" in execution_metadata:
            error_msg = execution_metadata["error"]
            feedback_parts.append(f"\nCompilation/Runtime Error:")
            feedback_parts.append(f"{error_msg}")
            
        if "traceback" in execution_metadata:
            traceback_msg = execution_metadata["traceback"]
            feedback_parts.append(f"\nDetailed traceback:")
            feedback_parts.append(f"{traceback_msg}")
            
        # Handle wrong answer cases
        if "error_message" in execution_metadata and execution_metadata["error_message"] == "Wrong Answer":
            feedback_parts.append(f"\nYour code executed but produced incorrect output:")
            
            if "inputs" in execution_metadata:
                feedback_parts.append(f"Input: {execution_metadata['inputs']}")
            if "expected" in execution_metadata:
                feedback_parts.append(f"Expected output: {execution_metadata['expected']}")
            if "output" in execution_metadata:
                feedback_parts.append(f"Your output: {execution_metadata['output']}")
        
        # If no specific error info but score is 0, provide generic feedback
        if score == 0 and not any(key in execution_metadata for key in ["error", "traceback", "error_message"]):
            feedback_parts.append(f"\nYour code did not produce the expected output.")
            # Try to show any available details
            if execution_metadata:
                feedback_parts.append(f"Debug info: {execution_metadata}")
                    
        # Add general guidance
        feedback_parts.append(f"\nPlease analyze the feedback and fix your code.")
        
        return "\n".join(feedback_parts)

    def process_turn(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata: List[CodeEnvironmentMetadata],
    ) -> Tuple[
        List[Dict[str, str]],  # observations
        List[float],           # rewards
        List[bool],            # terminateds
        List[Optional[List[str]]], # stop_strings
        List[Optional[CodeEnvironmentMetadata]], # next_metadata
    ]:
        """Process a batch of turns with multi-turn support."""
        
        # Extract the assistant's responses from the message history (same as original)
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        unittests = [prepare_tests(m) for m in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_unittests = chunk_list_to_workers(unittests, self.num_workers)

        # Process each chunk in parallel (same as original)
        futures = [
            self.workers[i].verify.remote(chunk, unittests_chunk, self.timeout)
            for i, (chunk, unittests_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_unittests)
            )
        ]

        results = ray.get(futures)

        # flatten the results (same as original)
        results = [item for sublist in results for item in sublist]
        
        # Process results with multi-turn logic
        observations = []
        rewards = []
        terminateds = []
        next_metadata = []
        
        for i, (score, execution_metadata) in enumerate(results):
            current_turn = metadata[i].get("current_turn", 0) + 1
            
            # Check max turns
            if current_turn > self.max_turns:
                content = "Maximum turns reached. Episode terminated."
                if self.role != "user":
                    content = f"<environment>\n{content}\n</environment>"
                
                observations.append({
                    "role": self.role,
                    "content": content
                })
                rewards.append(0.0)
                terminateds.append(True)
                next_metadata.append(None)
                continue
            
            # Apply turn-based reward multiplier if solution is correct
            if score > 0:
                # Apply penalty for each turn beyond the first
                turn_multiplier = self.turn_penalty ** (current_turn - 1)
                score = score * turn_multiplier
            
            # Create feedback
            feedback = self._format_error_feedback(execution_metadata, current_turn, score)
            if self.role != "user":
                feedback = f"<environment>\n{feedback}\n</environment>"
            
            observations.append({"role": self.role, "content": feedback})
            
            # Determine if episode should terminate
            is_correct = score > 0
            is_terminated = is_correct
            
            rewards.append(score)
            terminateds.append(is_terminated)
            
            if not is_terminated:
                # Update metadata for next turn
                new_metadata = metadata[i].copy()
                new_metadata["current_turn"] = current_turn
                next_metadata.append(new_metadata)
            else:
                next_metadata.append(None)

        next_stop_strings = [None] * len(message_log_batch)

        return observations, rewards, terminateds, next_stop_strings, next_metadata
