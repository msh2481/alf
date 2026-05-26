# Copyright (c) 2025 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Formatting utilities for nested data structures (nests).

Handles namedtuples, dicts, lists, tuples, tensors, and regular values.
"""
import torch
import numpy as np
from typing import Any


def format_nest(obj: Any, indent: int = 0, max_depth: int = 10) -> str:
    """
    Format a nested structure into a readable string representation.

    Args:
        obj: The object to format (can be nested)
        indent: Current indentation level
        max_depth: Maximum depth to recurse

    Returns:
        Formatted string representation
    """
    if max_depth <= 0:
        return " " * indent + "..."

    indent_str = " " * indent
    next_indent = indent + 2

    # Handle None and empty tuple (used as default in namedtuples)
    if obj is None or obj == ():
        return indent_str + "()"

    # Handle torch tensors
    if isinstance(obj, torch.Tensor):
        shape_str = ", ".join(str(s) for s in obj.shape)
        device_str = f" on {obj.device}" if obj.device.type != "cpu" else ""
        stats = ""
        if obj.numel() > 0 and obj.dtype in [
                torch.float32, torch.float64, torch.float16, torch.int32,
                torch.int64, torch.int16
        ]:
            try:
                mean_val = obj.float().mean().item()
                std_val = obj.float().std().item() if obj.numel() > 1 else 0.0
                stats = f"({mean_val:.2f} ± {std_val:.2f})"
            except:
                pass
        return indent_str + f"{obj.dtype}[{shape_str}]{stats}{device_str}"

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        shape_str = ", ".join(str(s) for s in obj.shape)
        stats = ""
        if obj.size > 0 and obj.dtype in [np.float32, np.float64, np.float16]:
            try:
                stats = f"({obj.mean():.2f} ± {obj.std():.2f})"
            except:
                pass
        return indent_str + f"numpy.{obj.dtype}[{shape_str}]{stats}"

    # Handle namedtuples
    if hasattr(obj, '_fields'):
        type_name = type(obj).__name__
        lines = [indent_str + f"{type_name}("]
        for field in obj._fields:
            value = getattr(obj, field)
            if value == ():
                lines.append(" " * next_indent + f"{field}: ()")
            else:
                lines.append(" " * next_indent + f"{field}:")
                formatted_value = format_nest(value, next_indent + 2,
                                              max_depth - 1)
                lines.append(formatted_value)
        lines.append(indent_str + ")")
        return "\n".join(lines)

    # Handle dictionaries
    if isinstance(obj, dict):
        if not obj:
            return indent_str + "{}"
        lines = [indent_str + "{"]
        for key, value in obj.items():
            lines.append(" " * next_indent + f"{key}:")
            formatted_value = format_nest(value, next_indent + 2,
                                          max_depth - 1)
            lines.append(formatted_value)
        lines.append(indent_str + "}")
        return "\n".join(lines)

    # Handle lists
    if isinstance(obj, list):
        if not obj:
            return indent_str + "[]"
        if len(obj) > 10:
            # For long lists, show first few and last few
            lines = [
                indent_str + f"[{len(obj)} items, showing first 3 and last 1]"
            ]
            for i in [0, 1, 2]:
                lines.append(" " * next_indent + f"[{i}]:")
                formatted_value = format_nest(obj[i], next_indent + 2,
                                              max_depth - 1)
                lines.append(formatted_value)
            lines.append(" " * next_indent + "...")
            lines.append(" " * next_indent + f"[{len(obj)-1}]:")
            formatted_value = format_nest(obj[-1], next_indent + 2,
                                          max_depth - 1)
            lines.append(formatted_value)
            return "\n".join(lines)
        else:
            lines = [indent_str + "["]
            for i, item in enumerate(obj):
                lines.append(" " * next_indent + f"[{i}]:")
                formatted_value = format_nest(item, next_indent + 2,
                                              max_depth - 1)
                lines.append(formatted_value)
            lines.append(indent_str + "]")
            return "\n".join(lines)

    # Handle tuples (non-named)
    if isinstance(obj, tuple):
        if not obj:
            return indent_str + "()"
        if len(obj) > 10:
            lines = [
                indent_str + f"({len(obj)} items, showing first 3 and last 1)"
            ]
            for i in [0, 1, 2]:
                formatted_value = format_nest(obj[i], next_indent,
                                              max_depth - 1)
                lines.append(formatted_value)
            lines.append(" " * next_indent + "...")
            formatted_value = format_nest(obj[-1], next_indent, max_depth - 1)
            lines.append(formatted_value)
            return "\n".join(lines)
        else:
            lines = [indent_str + "("]
            for item in obj:
                formatted_value = format_nest(item, next_indent, max_depth - 1)
                lines.append(formatted_value)
            lines.append(indent_str + ")")
            return "\n".join(lines)

    # Handle primitive types
    return indent_str + repr(obj)


def format_shape(obj: Any) -> str:
    """
    Get a compact shape/type summary of a nested structure.

    Useful for quick overview without full details.

    Args:
        obj: The object to summarize

    Returns:
        Compact shape summary string
    """
    if obj == () or obj is None:
        return "()"

    if isinstance(obj, torch.Tensor):
        shape_str = "x".join(str(s) for s in obj.shape)
        return f"Tensor({shape_str})"

    if isinstance(obj, np.ndarray):
        shape_str = "x".join(str(s) for s in obj.shape)
        return f"ndarray({shape_str})"

    if hasattr(obj, '_fields'):
        type_name = type(obj).__name__
        field_summaries = []
        for field in obj._fields:
            value = getattr(obj, field)
            if value != ():
                field_summaries.append(f"{field}: {format_shape(value)}")
        return f"{type_name}({', '.join(field_summaries)})"

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{k}: {format_shape(v)}" for k, v in list(obj.items())[:3]]
        if len(obj) > 3:
            items.append("...")
        return "{" + ", ".join(items) + "}"

    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]" if isinstance(obj, list) else "()"
        return f"[{len(obj)} items]"

    return str(type(obj).__name__)
