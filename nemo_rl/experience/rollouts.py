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

# Generate rollouts for arbitrary environments
# Supports multi-turn rollouts and many simultaneous environments (E.g. you can train on math, code, multi-turn games and more at once)

from typing import Any, Dict, List, Tuple, Optional, Union

import ray
import torch
import re
from transformers import AutoTokenizer

from nemo_rl.data.interfaces import (
    DatumSpec,
    FlatMessagesType,
    LLMMessageLogType,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
    message_log_to_flat_messages,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)


def generate_responses(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: AutoTokenizer,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
) -> Tuple[BatchedDataDict[DatumSpec], List[torch.Tensor], dict]:
    """Generate responses from policy."""
    # Add stop_strings to generation_input_data if present in the batch
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        # Ensure the key exists even if it's None, matching GenerationDatumSpec
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    # Generate responses
    generation_outputs = policy_generation.generate(
        generation_input_data, greedy=greedy
    )

    # Extract generated tokens
    generated_ids = []
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]
    for output_ids, input_length, total_length in zip(
        generation_outputs["output_ids"], input_lengths, unpadded_sequence_lengths
    ):
        generated_ids.append(output_ids[input_length:total_length])

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        message = {
            "role": "assistant",
            "content": text,
            "token_ids": generation_outputs["output_ids"][i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(message)

    metrics = {
        "mean_generation_length": (
            torch.sum(unpadded_sequence_lengths) - torch.sum(input_lengths)
        ).item()
        / len(unpadded_sequence_lengths),
        "max_seqlen": torch.max(unpadded_sequence_lengths).item(),
    }

    return batch, generated_ids, metrics


def calculate_rewards(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: Dict[str, EnvironmentInterface],
) -> EnvironmentReturn:
    """Calculate rewards for generated responses and get environment feedback.

    Args:
        batch: Batch containing message_log (LLMMessageLogType) with generated responses
        task_to_env: Dictionary mapping task names to their corresponding environments

    Returns:
        EnvironmentReturn namedtuple containing:
            - observations: List of observations from the environment for the next turn.
            - metadata: List of extracted metadata from the environment.
            - next_stop_strings: List of stop strings for the next generation step.
            - rewards: Tensor of rewards for the last turn.
            - terminateds: Tensor of booleans indicating if an episode ended naturally.
    """
    # Extract message logs for environment (most recent interaction)
    to_env = [
        get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
        for i in range(len(batch["message_log"]))
    ]
    task_names = batch["task_name"]

    # Group messages by task type
    task_groups = {}
    for i, task_name in enumerate(task_names):
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append((i, to_env[i]))

    # Calculate rewards for each task group concurrently
    futures = []
    future_to_indices = {}  # Map future to its corresponding indices
    for task_name, group in task_groups.items():
        if task_name not in task_to_env:
            raise ValueError(f"No environment found for task type: {task_name}")

        # Extract indices and messages for this group
        indices = [idx for idx, _ in group]
        messages = [msg for _, msg in group]

        # Get corresponding environment info
        env_info = [batch["extra_env_info"][i] for i in indices]

        # Submit task to environment and store future
        future = task_to_env[task_name].step.remote(messages, env_info)
        futures.append(future)
        future_to_indices[future] = indices

    results = ray.get(futures)
    all_rewards = []
    all_env_observations = []
    all_terminateds = []
    all_next_stop_strings = []
    all_metadata = []  # Store extracted metadata
    all_indices_order = []

    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        # Environment step returns: EnvironmentReturn
        env_observations, metadata, next_stop_strings, task_rewards, terminateds = (
            result
        )
        if next_stop_strings is None:
            next_stop_strings = [None] * len(task_rewards)

        # Store results with their original indices
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            all_rewards.append(task_rewards[i])
            all_env_observations.append(env_observations[i])
            all_terminateds.append(terminateds[i])
            all_next_stop_strings.append(next_stop_strings[i])
            all_metadata.append(metadata[i])

    # Sort results by original index to maintain order
    sorted_indices = sorted(
        range(len(all_indices_order)), key=lambda k: all_indices_order[k]
    )
    rewards = torch.tensor([all_rewards[i] for i in sorted_indices])
    env_observations = [all_env_observations[i] for i in sorted_indices]
    terminateds = torch.tensor([all_terminateds[i] for i in sorted_indices])
    next_stop_strings = [all_next_stop_strings[i] for i in sorted_indices]
    metadata = [all_metadata[i] for i in sorted_indices]  # Sort metadata

    return EnvironmentReturn(
        observations=env_observations,
        metadata=metadata,
        next_stop_strings=next_stop_strings,
        rewards=rewards,
        terminateds=terminateds,
    )


def extract_reasoning_traces(message_logs: List[List[Dict[str, Any]]]) -> List[Optional[str]]:
    """Extract reasoning traces from a batch of message logs.
    
    Args:
        message_logs: List of message logs, each containing assistant messages
        
    Returns:
        List of extracted reasoning traces, or None if no reasoning trace found
    """
    reasoning_traces = []
    
    for message_log in message_logs:
        # Find the last assistant message in each log
        for msg in reversed(message_log):
            if msg["role"] == "assistant":
                if "<think>" in msg["content"] and "</think>" in msg["content"]:
                    reasoning_match = re.search(r"<think>(.*?)</think>", msg["content"], re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                        reasoning_traces.append(reasoning)
                    else:
                        reasoning_traces.append(None)
                else:
                    reasoning_traces.append(None)
                break
        else:
            # No assistant message found
            reasoning_traces.append(None)
    
    return reasoning_traces


def generate_reasoning_summaries(
    reasoning_traces: List[Optional[str]],
    policy_generation: GenerationInterface,
    tokenizer: AutoTokenizer
) -> List[Optional[str]]:
    """Generate summaries for a batch of reasoning traces.
    
    Args:
        reasoning_traces: List of reasoning traces to summarize
        policy_generation: Generation interface for creating summaries
        tokenizer: Tokenizer for processing text
        
    Returns:
        List of reasoning summaries
    """
    # Filter out None values and keep track of indices
    valid_indices = []
    valid_traces = []
    
    for i, trace in enumerate(reasoning_traces):
        if trace is not None:
            valid_indices.append(i)
            valid_traces.append(trace)
    
    if not valid_traces:
        return [None] * len(reasoning_traces)
    
    # Create prompts for each valid reasoning trace
    summary_prompts = [
        f"""Summarize the following reasoning process in 2-3 sentences, maintaining the same style and tone. Only output the summary, nothing else:

{trace}

Summary:"""
        for trace in valid_traces
    ]
    
    # Tokenize all prompts
    tokenized_prompts = tokenizer(
        summary_prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    input_ids = tokenized_prompts["input_ids"].to(policy_generation.device)
    attention_mask = tokenized_prompts["attention_mask"].to(policy_generation.device)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    
    # Generate summaries in batch
    generation_input_data = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "stop_strings": [None] * len(valid_traces),
        }
    )
    
    generation_outputs = policy_generation.generate(generation_input_data, greedy=True)
    
    # Extract generated summaries
    output_ids = generation_outputs["output_ids"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]
    
    summaries = []
    for i, (output, input_length, total_length) in enumerate(
        zip(output_ids, input_lengths, unpadded_sequence_lengths)
    ):
        summary_ids = output[input_length:total_length]
        summary = tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
        summaries.append(summary)
    
    # Map summaries back to original indices
    result = [None] * len(reasoning_traces)
    for i, summary in zip(valid_indices, summaries):
        result[i] = summary
    
    return result


def strip_reasoning_from_assistant_messages(
    message_logs: List[List[Dict[str, Any]]],
    tokenizer: AutoTokenizer,
    policy_generation: Optional[GenerationInterface] = None
) -> List[Optional[str]]:
    """Strip <think>...</think> reasoning traces from assistant messages and re-tokenize.
    
    Also generates summaries of reasoning if policy_generation is provided.
    
    Args:
        message_logs: List of message logs to process in-place
        tokenizer: Tokenizer for re-tokenizing the stripped content
        policy_generation: Optional generation interface for creating summaries
        
    Returns:
        List of reasoning summaries for each message log
    """
    reasoning_summaries = [None] * len(message_logs)
    
    # First, extract all reasoning traces
    if policy_generation is not None:
        reasoning_traces = extract_reasoning_traces(message_logs)
        
        # Generate summaries in batch
        reasoning_summaries = generate_reasoning_summaries(
            reasoning_traces, policy_generation, tokenizer
        )
    
    # Now strip reasoning from all message logs
    for message_log in message_logs:
        for msg in message_log:
            if msg["role"] == "assistant" and "</think>" in msg["content"]:
                # Extract final response
                final_response = msg["content"].split("</think>")[-1].strip()
                
                # Update content
                msg["content"] = final_response
                
                # Re-tokenize with same parameters as environment observations
                msg["token_ids"] = tokenizer(
                    final_response, 
                    return_tensors="pt", 
                    add_special_tokens=False
                )["input_ids"][0]
                if len(msg["token_ids"]) == 0:
                    # if there is an empty message, the empty `token_ids` tensor ends up being in fp32,
                    # which causes `_validate_tensor_consistency` to fail. To fix this, we convert the
                    # empty tensor to int64.
                    msg["token_ids"] = msg["token_ids"].to(torch.int64)
    
    return reasoning_summaries


def run_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: AutoTokenizer,
    task_to_env: Dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> Tuple[BatchedDataDict[DatumSpec], Dict[str, Any]]:
    """Runs a multi-turn rollout loop, interacting with the environment.

    Args:
        policy_generation: The generation interface (policy).
        input_batch: The starting batch containing initial message logs.
        tokenizer: The tokenizer.
        task_to_env: Dictionary mapping task names to environment instances.
        max_rollout_turns: Maximum number of agent-environment interaction turns.
        max_seq_len: Maximum sequence length allowed.
        greedy: Whether to use greedy decoding.

    Returns:
        Tuple containing:
            - BatchedDataDict with the full interaction history and accumulated rewards
            - Dictionary of rollout metrics
    """
    current_batch = input_batch.copy()  # Work on a copy
    batch_size = len(current_batch["message_log"])
    active_indices = torch.arange(batch_size)
    total_rewards = torch.zeros(batch_size, dtype=torch.float32)

    # Initialize stop_strings from the initial batch if present
    current_stop_strings = current_batch.get("stop_strings", [None] * batch_size)

    # Tracking metrics for each sample
    sample_turn_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_assistant_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_env_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_terminated = torch.zeros(batch_size, dtype=torch.bool)
    sample_truncated = torch.zeros(batch_size, dtype=torch.bool)
    sample_max_turns_reached = torch.zeros(batch_size, dtype=torch.bool)

    # Tracking per-turn metrics
    total_gen_tokens_per_turn = []
    active_samples_per_turn = []

    for turn in range(max_rollout_turns):
        if len(active_indices) == 0:
            break

        active_samples_per_turn.append(len(active_indices))

        # Convert LLMMessageLogType to FlatMessagesType for generation
        active_batch = current_batch.select_indices(active_indices)
        active_stop_strings = [current_stop_strings[i] for i in active_indices.tolist()]

        # For turns after the first one, generate summaries of reasoning traces
        if turn > 0:
            # Extract reasoning traces from the current turn's responses
            reasoning_traces = extract_reasoning_traces(active_batch["message_log"])
            
            # Generate summaries for the reasoning traces
            reasoning_summaries = generate_reasoning_summaries(
                reasoning_traces, policy_generation, tokenizer
            )
            
            # Replace original reasoning traces with summaries in the message log
            for i, message_log in enumerate(active_batch["message_log"]):
                for msg in message_log:
                    if msg["role"] == "assistant" and "</think>" in msg["content"]:
                        if reasoning_summaries[i]:
                            # Replace original reasoning with summary
                            summary_content = f"<think>\n{reasoning_summaries[i]}\n</think>\n"
                            final_response = msg["content"].split("</think>")[-1].strip()
                            msg["content"] = summary_content + final_response
                            
                            # Re-tokenize with updated content
                            msg["token_ids"] = tokenizer(
                                msg["content"], 
                                return_tensors="pt", 
                                add_special_tokens=False
                            )["input_ids"][0]
                            if len(msg["token_ids"]) == 0:
                                msg["token_ids"] = msg["token_ids"].to(torch.int64)

        active_flat_messages: FlatMessagesType
        active_flat_messages, active_input_lengths = (
            batched_message_log_to_flat_message(
                active_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
            )
        )

        # Extract input_ids and lengths from the flat messages
        active_input_ids = active_flat_messages["token_ids"]

        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": active_input_ids,
                "input_lengths": active_input_lengths,
                "stop_strings": active_stop_strings,
            }
        )

        # generate_responses updates active_batch["message_log"] in-place
        active_batch, generated_ids, gen_metrics = generate_responses(
            policy_generation,
            generation_input_data,
            active_batch,
            tokenizer,
            active_input_lengths,
            greedy=greedy,
        )

        # Record token usage - assistant
        for i, global_idx in enumerate(active_indices.tolist()):
            sample_assistant_token_counts[global_idx] += len(generated_ids[i])
            sample_token_counts[global_idx] += len(generated_ids[i])

        # Track total generated tokens this turn
        total_gen_tokens_per_turn.append(sum(len(ids) for ids in generated_ids))

        # Calculate rewards and get environment feedback
        env_output: EnvironmentReturn = calculate_rewards(active_batch, task_to_env)

        total_rewards[active_indices] += env_output.rewards

        # Update message log for ALL active samples with env observation
        # This must happen AFTER reasoning processing but BEFORE filtering based on done flags
        truncation_mask = torch.zeros_like(env_output.terminateds, dtype=torch.bool)
        for i, global_idx in enumerate(active_indices.tolist()):
            env_obs_content = env_output.observations[i]["content"]
            
            # Tokenize the raw content from the environment
            tokenized_obs = tokenizer(
                env_obs_content, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            if len(tokenized_obs) == 0:
                tokenized_obs = tokenized_obs.to(torch.int64)

            # check if new message overflows max_seq_len
            if (
                len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i]
                >= max_seq_len
            ):
                # truncate
                tokenized_obs = tokenized_obs[
                    : max_seq_len - (len(generated_ids[i]) + active_input_lengths[i])
                ]
                truncation_mask[i] = True
                # Record truncation
                sample_truncated[active_indices[i]] = True

            tokenized_env_obs_message = {
                "role": env_output.observations[i]["role"],
                "content": env_obs_content,
                "token_ids": tokenized_obs,
            }
            current_batch["message_log"][global_idx].append(tokenized_env_obs_message)

            # Record token usage - environment
            sample_env_token_counts[global_idx] += len(tokenized_obs)
            sample_token_counts[global_idx] += len(tokenized_obs)

            # Increment turn count
            sample_turn_counts[global_idx] += 1

        # Determine done samples and update active set
        terminateds = env_output.terminateds.bool()
        done = truncation_mask | terminateds
        sample_terminated[active_indices] |= done

        # Update active indices for the next iteration
        active_indices_local_next = torch.where(~done)[0]
        active_indices = active_indices[active_indices_local_next]
        continuing_indices_global = active_indices  # Indices relative to original batch
        # Get next stop strings and infos corresponding to the indices that are *continuing*
        continuing_next_stops = [
            env_output.next_stop_strings[i] for i in active_indices_local_next.tolist()
        ]
        # Get metadata corresponding to continuing indices, using the correct field name
        continuing_metadata = [
            env_output.metadata[i] for i in active_indices_local_next.tolist()
        ]

        for i, global_idx in enumerate(continuing_indices_global.tolist()):
            # Update stop strings for the next turn
            current_stop_strings[global_idx] = continuing_next_stops[i]
            # Update metadata (extra_env_info) using info from environment
            if continuing_metadata[i] is not None:
                current_batch["extra_env_info"][global_idx] = continuing_metadata[i]

    # Record samples that reached max turns
    sample_max_turns_reached[active_indices] = True

    # Add total rewards to the final batch
    current_batch["total_reward"] = total_rewards

    # Calculate aggregate metrics
    rollout_metrics = {
        # Overall metrics
        "total_turns": int(sample_turn_counts.sum().item()),
        "avg_turns_per_sample": float(sample_turn_counts.float().mean().item()),
        "max_turns_per_sample": int(sample_turn_counts.max().item()),
        "natural_termination_rate": float(sample_terminated.float().mean().item()),
        "truncation_rate": float(sample_truncated.float().mean().item()),
        "max_turns_reached_rate": float(sample_max_turns_reached.float().mean().item()),
        # Token usage metrics
        "mean_gen_tokens_per_sample": float(sample_token_counts.float().mean().item()),
        "min_gen_tokens_per_sample": float(sample_token_counts.float().min().item()),
        "max_gen_tokens_per_sample": float(sample_token_counts.float().max().item()),
        "mean_env_tokens_per_sample": float(
            sample_env_token_counts.float().mean().item()
        ),
    }
    return current_batch, rollout_metrics
