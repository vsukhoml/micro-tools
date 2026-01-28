#!/usr/bin/env python3
import sys
import os
import argparse
import struct
from io import BytesIO
from collections import defaultdict

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
    from elftools.elf.enums import ENUM_ST_SHNDX
    from elftools.elf.relocation import RelocationSection
except ImportError:
    print("Error: 'pyelftools' is required. Please install it: pip install pyelftools", file=sys.stderr)
    sys.exit(1)

class ArReader:
    """
    Reads a GNU/SysV 'ar' archive directly in memory.
    Handles '/' termination and '//' Long Filename tables.
    """
    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, 'rb')
        self.long_names = None

    def __enter__(self):
        # Check Magic Header
        if self.f.read(8) != b'!<arch>\n':
            raise ValueError("Not a valid ar archive")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def __iter__(self):
        while True:
            # Read 60-byte file header
            header = self.f.read(60)
            if not header or len(header) < 60:
                break

            # Parse Header (ASCII)
            # 0-15: Name
            # 48-57: Size
            try:
                raw_name = header[0:16].strip()
                size = int(header[48:58].strip())
            except ValueError:
                break

            # Read content
            content = self.f.read(size)

            # Alignment padding (2-byte boundary)
            if size % 2 == 1:
                self.f.read(1)

            name_str = raw_name.decode('utf-8', errors='ignore')

            # 1. Symbol Table (Special Entry)
            if name_str == '/':
                continue # Skip symbol table

            # 2. Long Filename Table (Special Entry)
            elif name_str == '//':
                self.long_names = content
                continue

            # 3. Reference to Long Filename (e.g., /18)
            elif name_str.startswith('/') and name_str[1:].isdigit():
                if self.long_names is None:
                    print(f"Warning: Found long name ref {name_str} before table loaded.", file=sys.stderr)
                    final_name = name_str
                else:
                    offset = int(name_str[1:])
                    # GNU ar format: names are terminated by '/' followed by newline
                    end = self.long_names.find(b'/', offset)
                    if end != -1:
                        final_name = self.long_names[offset:end].decode('ascii')
                    else:
                        final_name = name_str # Fallback

            # 4. Standard Name (e.g., rsa.o/)
            elif name_str.endswith('/'):
                final_name = name_str[:-1]

            else:
                final_name = name_str

            yield final_name, content

class ArchiveAnalyzer:
    def __init__(self):
        # Global map of defined symbols: name -> (filepath, section_index, size)
        self.global_symbols = {}
        # Map of local symbols (static): (filepath, name) -> (filepath, section_index, size)
        self.local_symbols = {}
        # Dependencies: (filepath, section_index) -> Set of dependencies
        self.section_deps = defaultdict(set)
        # Section sizes: (filepath, section_index) -> size in bytes
        self.section_sizes = {}
        # Human readable map for reporting: (filepath, section_index) -> section_name
        self.section_names = {}

    def process_archive(self, archive_path):
        """Reads archive and processes ELF objects in memory."""

        # We track occurrence of filenames to handle duplicates (rsa.o, rsa.o)
        filename_counts = defaultdict(int)

        with ArReader(archive_path) as ar:
            for filename, content in ar:
                # Create a unique ID for this object file
                count = filename_counts[filename]
                unique_obj_id = f"{filename}" if count == 0 else f"{filename} ({count})"
                filename_counts[filename] += 1

                # Check if it looks like an ELF file (starts with \x7fELF)
                if content.startswith(b'\x7fELF'):
                    try:
                        self._analyze_object(unique_obj_id, content)
                    except Exception as e:
                        print(f"Warning: Failed to parse {unique_obj_id}: {e}", file=sys.stderr)


    def _analyze_object(self, filepath, content):
        # Wrap bytes in BytesIO so ELFFile can treat it as a stream
        stream = BytesIO(content)
        elf = ELFFile(stream)

        # 1. Map Section Indices to Properties
        for idx, section in enumerate(elf.iter_sections()):
            self.section_sizes[(filepath, idx)] = section['sh_size']
            self.section_names[(filepath, idx)] = section.name

        # 2. Process Symbol Table
        symtab = elf.get_section_by_name('.symtab')
        if not symtab:
            return

        # Store symbols to help resolve relocations later
        internal_sym_map = {}

        for i, sym in enumerate(symtab.iter_symbols()):
            internal_sym_map[i] = sym

            if sym['st_shndx'] == 'SHN_UNDEF':
                continue

            if isinstance(sym['st_shndx'], int):
                sec_idx = sym['st_shndx']
                location = (filepath, sec_idx, sym['st_size'])

                if sym['st_info']['bind'] == 'STB_GLOBAL':
                    if sym.name not in self.global_symbols:
                        self.global_symbols[sym.name] = location
                else:
                    self.local_symbols[(filepath, sym.name)] = location

        # 3. Process Relocations
        for section in elf.iter_sections():
            if not isinstance(section, RelocationSection):
                continue

            target_sec_idx = section['sh_info']
            origin_node = (filepath, target_sec_idx)

            for rel in section.iter_relocations():
                sym_idx = rel['r_info_sym']
                sym = internal_sym_map.get(sym_idx)

                if not sym:
                    continue
                # Case A: Relocation against a Section (e.g., .rodata) -> Local Dependency
                if sym['st_info']['type'] == 'STT_SECTION':
                    dep_sec_idx = sym['st_shndx']
                    self.section_deps[origin_node].add((filepath, dep_sec_idx))

                # Case B: Relocation against a Defined Symbol -> Local or Global Dependency
                elif isinstance(sym['st_shndx'], int):
                    if sym['st_info']['bind'] == 'STB_LOCAL':
                        self.section_deps[origin_node].add((filepath, sym['st_shndx']))
                    else:
                        self.section_deps[origin_node].add(sym.name)
                # Case C: Relocation against Undefined Symbol -> External Dependency
                elif sym['st_shndx'] == 'SHN_UNDEF':
                    self.section_deps[origin_node].add(sym.name)

    def calculate_size(self, root_symbols):
        """
        Calculates total size starting from root_symbols list.
        """
        visited_sections = set()
        queue = []

        print(f"--- Resolving {len(root_symbols)} Root Symbols ---")
        for sym_name in root_symbols:
            # Check Global Symbols
            if sym_name in self.global_symbols:
                loc = self.global_symbols[sym_name]
                node = (loc[0], loc[1])
                print(f"[FOUND] {sym_name} -> {os.path.basename(loc[0])}:{self.section_names.get(node)}")
                queue.append(node)
                continue

            # Check Local Symbols (less likely for roots, but possible)
            found = False
            for (fp, name), loc in self.local_symbols.items():
                if name == sym_name:
                    node = (loc[0], loc[1])
                    print(f"[FOUND LOCAL] {sym_name} -> {os.path.basename(loc[0])}:{self.section_names.get(node)}")
                    queue.append(node)
                    found = True
                    break

            if not found:
                print(f"[MISSING] Could not find definition for symbol: {sym_name}")

        print(f"\n--- Processing Dependencies ---")

        while queue:
            current_node = queue.pop(0)
            if current_node in visited_sections:
                continue

            visited_sections.add(current_node)
            deps = self.section_deps.get(current_node, [])

            for dep in deps:
                next_node = None

                if isinstance(dep, tuple):
                    next_node = dep
                elif isinstance(dep, str):
                    if dep in self.global_symbols:
                        loc = self.global_symbols[dep]
                        next_node = (loc[0], loc[1])

                if next_node and next_node not in visited_sections:
                    queue.append(next_node)

        total_size = 0
        print(f"\n--- Included Sections ---")
        print(f"{'Size (Bytes)':<12} | {'Section Name':<40} | {'File'}")
        print("-" * 80)

        sorted_nodes = sorted(list(visited_sections), key=lambda x: (os.path.basename(x[0]), x[1]))

        for node in sorted_nodes:
            size = self.section_sizes.get(node, 0)
            name = self.section_names.get(node, "UNKNOWN")
            filename = os.path.basename(node[0])

            if size > 0:
                print(f"{size:<12} | {name:<40} | {filename}")
                total_size += size

        return total_size

def main():
    parser = argparse.ArgumentParser(description="Calculate ELF archive size by symbol dependencies.")
    parser.add_argument("archive", help="Path to the .a library archive")
    parser.add_argument("symbols", nargs='+', help="List of root symbols OR file paths prefixed with @ (e.g., @symbols.txt)")

    args = parser.parse_args()

    if not os.path.exists(args.archive):
        print(f"Error: Archive {args.archive} not found.")
        sys.exit(1)

    root_symbols = []

    for item in args.symbols:
        if item.startswith('@'):
            file_path = item[1:] # Remove the @
            if not os.path.exists(file_path):
                print(f"Error: Symbol file '{file_path}' not found.", file=sys.stderr)
                sys.exit(1)

            try:
                with open(file_path, 'r') as f:
                    # Read lines, strip whitespace, filter empty lines
                    file_symbols = [line.strip() for line in f if line.strip()]
                    root_symbols.extend(file_symbols)
                    print(f"Loaded {len(file_symbols)} symbols from {file_path}")
            except Exception as e:
                print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            root_symbols.append(item)

    if not root_symbols:
        print("Error: No symbols provided to analyze.", file=sys.stderr)
        sys.exit(1)

    analyzer = ArchiveAnalyzer()
    print(f"Scanning archive: {args.archive}...")
    analyzer.process_archive(args.archive)

    total = analyzer.calculate_size(root_symbols)

    print("-" * 80)
    print(f"Total Static Size: {total} bytes")

if __name__ == "__main__":
    main()
