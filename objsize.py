#!/usr/bin/env python3
import sys
import argparse

def calculate_section_size(objdump_output):
    """
    Calculates the total size of sections starting with .text and .rodata.

    Args:
        objdump_output: A string containing the output of 'objdump -h'.

    Returns:
        The total size of the specified sections in bytes.
    """
    total_size = 0
    for line in objdump_output.strip().split('\n'):
        parts = line.split()
        # Check if the line looks like a section header (starts with an index)
        if len(parts) > 2 and parts[0].isdigit():
            section_name = parts[1]
            if section_name.startswith('.text') or section_name.startswith('.rodata'):
                try:
                    section_size = int(parts[2], 16)
                    print(section_name, section_size)
                    total_size += section_size
                except (ValueError, IndexError):
                    # Ignore lines that don't parse correctly
                    continue
    return total_size

def main():
    """
    Main function to parse command-line arguments and process the file.
    """
    parser = argparse.ArgumentParser(
        description="Calculate the total size of .text and .rodata sections from 'objdump -h' output."
    )
    parser.add_argument(
        "filename",
        help="The path to the file containing the objdump output."
    )
    args = parser.parse_args()

    try:
        with open(args.filename, 'r') as f:
            objdump_content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.filename}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    total_size = calculate_section_size(objdump_content)
    print(f"Total size of .text and .rodata sections: {total_size} bytes")

if __name__ == "__main__":
    main()
