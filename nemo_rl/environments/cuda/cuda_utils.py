"""
Complete extraction of KernelBench utilities.
"""
import re
import os
from typing import List, Optional, Dict, Any

def extract_python_code(text):
    """
    Extract python code from model output
    COMPLETE extraction from KernelBench utils.py
    """
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches) if matches else ""

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
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

def extract_last_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract last code block from model output, specified by code_language_type
    COMPLETE extraction from KernelBench utils.py - no changes
    """
    trimmed = output_string.strip()

    # Find all matches of code blocks
    code_matches = re.finditer(r"```(.*?)```", trimmed, re.DOTALL)
    
    # Get the last match by converting to list and taking the last element
    matches_list = list(code_matches)
    if matches_list:
        last_match = matches_list[-1]
        code = last_match.group(1).strip()

        # Remove language type headers
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()

        return code
    
    return None

def extract_code_blocks(text, code_language_types: list[str]) -> str:
    '''
    Extract all code blocks from text, combine them to return as a single string
    COMPLETE extraction from KernelBench utils.py - no changes
    '''
    pattern = r'```.*?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    # Combine all code blocks and remove language type headers
    combined_code = []
    for match in matches:
        code = match.strip()
        # Remove any language type headers
        for lang_type in code_language_types:
            if code.startswith(lang_type):
                code = code[len(lang_type):].strip()
        combined_code.append(code)
    
    return " \n ".join(combined_code) if combined_code else ""

def extract_all_cuda_sources(file_content: str) -> list[str]:
    """
    Extract all CUDA sources wrapped in triple quotes.
    COMPLETE extraction from KernelBench analysis.py
    
    Returns:
        list[str]: List of all extracted CUDA source code blocks
    """
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*"""(.*?)"""'
    matches = re.findall(pattern, file_content, re.DOTALL)
    return [match.strip() for match in matches]

def set_gpu_arch(gpu_arch: str):
    """
    Set GPU architecture for CUDA compilation
    COMPLETE extraction from KernelBench utils.py
    """
    os.environ['TORCH_CUDA_ARCH_LIST'] = gpu_arch

def read_file(file_path: str) -> str:
    """
    Read file contents
    COMPLETE extraction from KernelBench utils.py
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()