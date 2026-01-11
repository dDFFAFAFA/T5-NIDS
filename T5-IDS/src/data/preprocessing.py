"""Bytes-to-hex preprocessing utilities."""

from typing import Iterable, List


def bytes_to_hex(byte_array: Iterable[Iterable[int]], format_type: str = "every4") -> List[str]:
    """Convert payload bytes into hex strings with optional spacing."""
    hex_strings = []
    for row in byte_array:
        hex_str = "".join(f"{int(b):02x}" for b in row)
        if format_type == "every4":
            hex_str = " ".join(hex_str[i:i + 4] for i in range(0, len(hex_str), 4))
        elif format_type == "every2":
            hex_str = " ".join(hex_str[i:i + 2] for i in range(0, len(hex_str), 2))
        elif format_type == "noSpace":
            hex_str = hex_str
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
        hex_strings.append(hex_str.strip())
    return hex_strings
