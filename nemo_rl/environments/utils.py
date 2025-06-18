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
from typing import Any, List
import re

def chunk_list_to_workers(to_chunk: List[Any], num_workers: int) -> List[List[Any]]:
    """Chunk a list into a list of lists, where each sublist is assigned to a worker. Keeps ordering of elements.

    If the list is not divisible by the number of workers, the last worker may have fewer elements.
    If there are more workers than elements, the first len(list) workers will have a single element each,
    and the remaining workers will have empty lists.

    Args:
        list: The list to be chunked.
        num_workers: The number of workers to distribute the list to.

    Returns:
        A list of lists, where each sublist contains elements assigned to a worker.

    Examples:
    ```{doctest}
    >>> from nemo_rl.environments.utils import chunk_list_to_workers
    >>> chunk_list_to_workers([1, 2, 3, 4, 5], 3)
    [[1, 2], [3, 4], [5]]
    ```
    """
    if not to_chunk:
        return [[] for _ in range(num_workers)]

    # Handle case where we have more workers than elements
    if len(to_chunk) <= num_workers:
        result = [[item] for item in to_chunk]
        # Add empty lists for remaining workers
        result.extend([[] for _ in range(num_workers - len(to_chunk))])
        return result

    # Calculate chunk size (ceiling division to ensure all elements are covered)
    chunk_size = (len(to_chunk) + num_workers - 1) // num_workers

    # Create chunks
    chunks = []
    for i in range(0, len(to_chunk), chunk_size):
        chunks.append(to_chunk[i : i + chunk_size])

    # If we somehow ended up with more chunks than workers (shouldn't happen with ceiling division)
    # merge the last chunks
    if len(chunks) > num_workers:
        chunks[num_workers - 1 :] = [sum(chunks[num_workers - 1 :], [])]

    return chunks


def extract_answer_from_box(string):
    """Extract Answer String from \\boxed expression."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None

# Taken from: https://github.com/ScalingIntelligence/KernelBench/blob/main/src/utils.py#L478
def extract_code(output_string: str, code_language_types: list[str] = ["python"]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    COMPLETE extraction from KernelBench utils.py - no changes
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None