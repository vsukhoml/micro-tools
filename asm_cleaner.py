#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Clang Assembly Cleaner.

Features:
1. Interleaves source code comments by resolving .loc directives to actual file content.
2. Collapses consecutive data directives (e.g., .short 0, .short 1 -> .short 0, 1).
3. Strips DWARF debug sections (.debug_*) entirely.
4. Removes temporary labels and metadata noise.
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Pattern, TextIO

# --- Constants & Regex ---

# Directives to skip entirely
SKIP_PREFIXES = (
    ".cfi_",
    ".cfi_startproc", ".cfi_endproc", ".cfi_def_cfa", ".cfi_offset",
    ".cfi_restore", ".cfi_undefined", ".cfi_same_value", ".cfi_window_save",
    ".cfi_register", ".cfi_def_cfa_offset", ".cfi_def_cfa_register"
)

# Regex to match temporary labels (e.g., .Ltmp123:)
RE_TEMP_LABEL = re.compile(r"^\s*\.Ltmp\d+:$")

# Regex to parse .file directives
# Format: .file <num> "directory" "filename" ... OR .file <num> "filename" ...
# Captures: 1=num, 2=quoted_string_1, 3=optional_quoted_string_2
RE_FILE_DIR = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"(?:\s+"([^"]+)")?')

# Regex to parse .loc directives
# Format: .loc <filenum> <linenum> <colnum> ...
RE_LOC = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)')

# Regex to identify data directives that should be merged
# Supported: .byte, .short, .long, .int, .quad, .word
RE_DATA_DIRECTIVE = re.compile(r'^\s*\.(byte|short|long|int|quad|word)\s+([^#\n]+)')

# --- Classes ---

class SourceManager:
    """Manages file mappings and source code retrieval."""
    def __init__(self):
        self.files: Dict[int, str] = {}
        self.cache: Dict[str, List[str]] = {}

    def parse_file_directive(self, line: str) -> None:
        """Parses a .file line and updates the file mapping."""
        # We use a regex to extract strings to handle quotes safely
        match = RE_FILE_DIR.match(line)
        if match:
            file_num = int(match.group(1))
            part1 = match.group(2)
            part2 = match.group(3)

            if part2:
                # Format: .file N "dir" "filename"
                full_path = os.path.join(part1, part2)
            else:
                # Format: .file N "filename"
                full_path = part1

            self.files[file_num] = full_path

    def get_source_line(self, file_num: int, line_num: int, col_num: int) -> Optional[str]:
        """
        Retrieves the line from source file, adding a column marker.
        Returns None if file/line not found or line_num is 0.
        """
        if line_num <= 0:
            return None

        path = self.files.get(file_num)
        if not path:
            return f"<unknown_file_ref_{file_num}>"

        # Try to load file into cache
        if path not in self.cache:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        self.cache[path] = f.readlines()
                except Exception:
                    self.cache[path] = [] # Mark as failed/empty
            else:
                self.cache[path] = []

        lines = self.cache[path]
        if line_num - 1 < len(lines):
            raw_content = lines[line_num - 1].rstrip('\n')

            # Insert column delimiter if col_num > 0
            # DWARF column is 1-based.
            if col_num > 1 and col_num <= len(raw_content) + 1:
                idx = col_num - 1
                # Using a subtle unicode marker '│', '‸' or standard '^'
                # User requested a delimiter.
                marked_content = raw_content[:idx] + "‸" + raw_content[idx:]
                return marked_content

            return raw_content

        return None


class DataCoalescer:
    """Buffers consecutive data directives to merge them."""
    def __init__(self):
        self.buffer: List[str] = []
        self.current_type: Optional[str] = None
        self.indent: str = "\t"

    def flush(self) -> List[str]:
        if not self.buffer:
            return []

        # Join values with commas
        merged_values = ",".join(self.buffer)
        output = [f"{self.indent}.{self.current_type}\t{merged_values}"]

        self.buffer = []
        self.current_type = None
        return output

    def process(self, line: str) -> bool:
        """
        Returns True if line was consumed (buffered).
        Returns False if line is not a mergeable data directive.
        """
        match = RE_DATA_DIRECTIVE.match(line)
        if match:
            dtype = match.group(1)
            val_str = match.group(2).strip()

            # Determine indentation from original line
            current_indent = line[:line.find('.')]

            if self.current_type is None:
                self.current_type = dtype
                self.indent = current_indent
                self.buffer.append(val_str)
                return True
            elif self.current_type == dtype:
                self.buffer.append(val_str)
                return True
            else:
                # Different type: flush current, then start new
                return False

        return False


# --- Main Processing ---

def process_assembly(input_stream: TextIO) -> None:
    source_mgr = SourceManager()
    data_buffer = DataCoalescer()

    in_debug_section = False

    for line in input_stream:
        raw_line = line.rstrip('\n')
        stripped = raw_line.strip()

        # 1. Debug Section Stripping
        # Check for section change
        if stripped.startswith(".section"):
            if ".debug_" in stripped:
                in_debug_section = True
            else:
                in_debug_section = False

        # If inside a debug section, skip everything until next section directive
        if in_debug_section:
            continue

        # 2. Flush data buffer if we hit a label or unrelated instruction
        # (We check logic inside data_buffer.process later, but labels definitely break flow)
        if stripped.endswith(':'):
             for flushed in data_buffer.flush():
                 print(flushed)

        # 3. Filter Temporary Labels
        if RE_TEMP_LABEL.match(raw_line):
            continue

        # 4. Handle .file directives
        if stripped.startswith(".file"):
            source_mgr.parse_file_directive(raw_line)
            print(raw_line) # Keep .file in output for reference
            continue

        # 5. Handle .loc directives (The Core Logic)
        loc_match = RE_LOC.match(raw_line)
        if loc_match:
            # Always flush data buffer before a source line comment
            for flushed in data_buffer.flush():
                print(flushed)

            f_num = int(loc_match.group(1))
            l_num = int(loc_match.group(2))
            c_num = int(loc_match.group(3))

            src_content = source_mgr.get_source_line(f_num, l_num, c_num)

            # Replace .loc line with source content if valid
            if src_content:
                # Match indentation of the original .loc for neatness, or default to tab
                # indent = raw_line[:raw_line.find('.loc')] if '.loc' in raw_line else "\t"
                #print(f"{indent}# {src_content}")
                print(f";{src_content}")

            # If line invalid (0 0 0), we simply remove it (print nothing)
            continue

        # 6. Skip CFI and other specific noise
        if stripped.startswith(SKIP_PREFIXES):
            continue

        # 7. Comment-only lines removal
        if not stripped or (stripped.startswith("#") and not stripped.startswith("# --")):
            # We explicitly keep "# --" as LLVM often uses it for function delimiters
            continue

        # 8. Data Directive Collapsing
        # Try to add to buffer
        if data_buffer.process(raw_line):
            continue

        # If not buffered, it might be a type switch or non-data line.
        # Flush existing buffer first.
        for flushed in data_buffer.flush():
            print(flushed)

        # If the line was a data directive but of a DIFFERENT type (causing flush),
        # we must try to process it again with the empty buffer.
        if RE_DATA_DIRECTIVE.match(raw_line):
             if data_buffer.process(raw_line):
                 continue

        # 9. Print regular line
        # Clean inline comments like "# 0x1" from instructions if needed?
        # User example showed ".short 0 # 0x0" -> ".short 0".
        # We did that in buffer. For regular instructions, usually we keep comments.
        print(raw_line)

    # Final flush
    for flushed in data_buffer.flush():
        print(flushed)

def main():
    parser = argparse.ArgumentParser(description="Clean and annotate Clang assembly.")
    parser.add_argument('input_file', nargs='?', default='-',
                        help="Input file (default: stdin)")

    args = parser.parse_args()

    if args.input_file == '-':
        process_assembly(sys.stdin)
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            process_assembly(f)

if __name__ == "__main__":
    main()
