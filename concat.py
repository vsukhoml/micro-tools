#!/usr/bin/env python3

# Use example: concat "./**/*" file_content.md

import argparse
import fnmatch
import os
import sys
import re
from pathlib import Path

# --- Configuration: Map file extensions to language tags ---
# Add more mappings as needed, 'None' means ignore this file
LANGUAGE_MAP = {
    '.lock': None,
    '.toml': None,
    '.bp': None,
    '.gitignore': None,
    '.hash': None,
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.html': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.java': 'java',
    '.kt': 'kotlin',
    '.rs': 'rust',
    '.go': 'go',
    '.c': 'c',
    '.cpp': 'cpp',
    '.cs': 'csharp',
    '.swift': 'swift',
    '.php': 'php',
    '.rb': 'ruby',
    '.pl': 'perl',
    '.sh': 'bash',
    '.zsh': 'zsh',
    '.sql': 'sql',
    '.md': 'markdown',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.aidl': 'aidl',
    '.xml': 'xml',
    '.dockerfile': 'dockerfile',
    'dockerfile': 'dockerfile', # Handle extensionless Dockerfiles if named 'Dockerfile'
    '.env': 'plaintext',       # Treat as plain text
    '.txt': 'plaintext',
    # Add other extensions and their corresponding language identifiers
}

def find_files(file_mask):
    """
    Finds files matching a glob-style mask using os.walk for efficiency.

    Args:
        file_mask (str): The file pattern to match (e.g., 'data/**/*.csv').

    Returns:
        list: A list of file paths that match the mask.
    """
    # If the mask has no wildcards, it's a simple path. Check if it's a file.
    if not any(char in file_mask for char in '*?['):
        if os.path.isfile(file_mask):
            return [file_mask]
        return []

    matched_files = []

    # --- Optimization: Find the optimal starting directory for the walk ---
    # We want to start walking from the deepest directory prefix in the mask
    # that does *not* contain any wildcards. This avoids scanning irrelevant directories.
    root_dir = os.path.dirname(file_mask)

    # Walk up the path components until we find a part with no wildcards or hit the top.
    while root_dir and any(char in os.path.basename(root_dir) for char in '*?['):
        root_dir = os.path.dirname(root_dir)

    # If the mask was relative (e.g., 'data*/*.txt'), root_dir can become an empty string.
    # In this case, the walk should start from the current directory ('.').
    if not root_dir:
        root_dir = '.'

    # If the calculated base path doesn't exist, no files can match.
    if not os.path.isdir(root_dir):
        return []

    # --- Walk the directory tree and match files ---
    for current_dir, _, files_in_dir in os.walk(root_dir, followlinks=False):
        # Skip hidden directories
        if "./." in current_dir:
            continue

        for filename in files_in_dir:
            # Construct the full path of the file.
            full_path = os.path.join(current_dir, filename)
            # Use fnmatch to check if the full path matches the original mask.
            if fnmatch.fnmatch(full_path, file_mask):
                matched_files.append(full_path)

    return matched_files

def get_language_tag(filepath: Path) -> str | None:
    """Determine the language tag based on file extension."""

    # Return `None` for files which are not needed
    if filepath.name in ['OWNERS', 'TEST_MAPPING', 'NOTICE', 'LICENSE', "PREUPLOAD.cfg"]:
        return None

    # Handle special case for extensionless Dockerfile
    filename = filepath.name.lower()
    if filename == 'dockerfile':
        return 'dockerfile'
    elif filename.startswith("build"):
        return "BUILD"
    elif filename.startswith("workspace"):
        return "WORKSPACE"
    elif filename.startswith("module"):
        return "MODULE"

    # Handle files starting with '.' like .gitignore
    if filepath.name.startswith('.') and filepath.name in LANGUAGE_MAP:
        return LANGUAGE_MAP.get(filepath.name, '')

    # General case based on suffix
    return LANGUAGE_MAP.get(filepath.suffix.lower(), filepath.suffix.lower())

def clean_starlark_content(content):
    # 1. Remove comments
    # This regex is simplified and might not handle all edge cases (e.g., '#' inside strings)
    # but works for most typical Bazel files.
    content = re.sub(r'#.*', '', content)

    # 2. Normalize excessive whitespace (keep single blank lines)
    # Replace 3 or more newlines with just 2 (i.e., one blank line)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 3. Strip leading/trailing whitespace from the whole content
    content = content.strip()

    return content

def read_input_file(file_path, file_is_bazel):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw_content = f.read()
        if file_is_bazel:
            raw_content = clean_starlark_content(raw_content)
        return raw_content

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate files matching a mask into a single output file, "
                    "tagging each file's content.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file_mask",
        help="File mask to match files (e.g., 'src/**/*.py', '*.rs', 'docs/*.md'). "
             "Use quotes if the mask contains wildcards."
    )
    parser.add_argument(
        "output_file",
        help="Name of the file to write the concatenated content to."
    )

    args = parser.parse_args()

    try:
        matched_files = find_files(args.file_mask)

        if not matched_files:
            print(f"Warning: No files found matching mask '{args.file_mask}'", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(matched_files)} file(s) matching '{args.file_mask}'.")
        print(f"Writing concatenated output to '{args.output_file}'...")

        # Open the output file for writing
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for filepath_str in sorted(matched_files): # Sort for consistent order
                filepath = Path(filepath_str)
                language = get_language_tag(filepath)
                file_is_bazel = language in ('BUILD', 'WORKSPACE', 'MODULE', 'Starlark', 'BazelRC')

                if language is None:
                    print(f"  Skipping: {filepath_str}")
                    continue

                print(f"  Processing: {filepath_str} (language: {language or 'unknown'})")
                try:
                    file_contents = read_input_file(filepath, file_is_bazel)
                    # Write the start tag
                    if file_is_bazel:
                        # Convert file path into Bazel representation
                        bazel_path = filepath_str.lstrip("./")
                        if bazel_path.count('/') > 0:
                            parts = bazel_path.rsplit('/', 1)
                            filepath_str = f"//{parts[0]}:{parts[1]}"
                        else:
                            filepath_str = f"//:{bazel_path}"

                    # Write the start tag
                    outfile.write(f"**File:** `{filepath_str}`\n")
                    # Write the code block opening tag
                    outfile.write(f"```{language}\n{file_contents}\n```\n\n---\n\n")

                except FileNotFoundError:
                    print(f"  Error: File not found (skipped): {filepath_str}", file=sys.stderr)
                except IOError as e:
                    print(f"  Error reading file {filepath_str}: {e} (skipped)", file=sys.stderr)
                except Exception as e:
                    print(f"  An unexpected error occurred processing {filepath_str}: {e} (skipped)", file=sys.stderr)

        print(f"Successfully created '{args.output_file}'.")

    except IOError as e:
        print(f"Error opening or writing to output file {args.output_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
