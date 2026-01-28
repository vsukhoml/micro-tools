#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Classifier utilizing Google Gemini API.
Classifies PDF files into a directory-based taxonomy.

Usage:
    python classify_pdf.py /path/to/incoming /path/to/library

Dependencies:
    pip install google-genai pypdf
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

# Import the specific error class from the SDK
from google import genai
from google.genai import errors, types

# --- Third Party Libraries ---
from pypdf import PdfReader

# --- Configuration Constants ---
TAXONOMY_FILENAME = "taxonomy.md"
LLM_MODEL = 'gemini-2.5-flash'
BATCH_SIZE = 32
CONTENT_PREVIEW_LENGTH = 2000  # Characters
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
SUPPORTED_EXTS_TEXT = {'.txt', '.md', '.htm', '.html'}
SUPPORTED_EXTS_PDF = {'.pdf'}
SUPPORTED_EXTS_NO_CONTENT = {'.djvu', '.epub', '.doc', '.docx', '.ppt', '.pptx', '.mobi', '.rtf', '.fb2', '.odt', '.mht', '.chm'}
ALL_SUPPORTED_EXTS = SUPPORTED_EXTS_TEXT | SUPPORTED_EXTS_PDF | SUPPORTED_EXTS_NO_CONTENT

# Configure Logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

T = TypeVar("T")

{'error': {'code': 429,  'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help',
   'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]},
   {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '59s'}]}}
# ['error']. `violations`[]  @type

def retry_on_429(max_retries: int = 5, default_delay: float = 5.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to handle Gemini API 429 Resource Exhausted errors.
    Parses 'retryDelay' from the server response if available.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except errors.ClientError as e:
                    # Check for 429 Resource Exhausted, 503 - model overloaded
                    if e.code != 429 and e.code != 503:
                        raise e

                    if retries >= max_retries:
                        logger.error(f"Max retries exceeded for {e.code} error.")
                        raise e

                    retries += 1
                    wait_time = default_delay * (2 ** (retries - 1)) # Default exponential backoff

                    # Attempt to extract precise retryDelay from exception details
                    # The error object structure is usually: e.details -> list of dicts
                    # We look for the one with 'retryDelay'
                    try:
                        # Inspect the raw error details if accessible
                        # Note: The SDK maps the JSON error to the exception attributes.
                        # We iterate over details to find the RetryInfo protobuf-like dict.
                        if hasattr(e, 'details') and e.details:
                            # Gemini specific logic to get the requested delay
                            error = e.details.get("error", {})
                            details = error.get('details', [])
                            for detail in details:
                                # We look for the specific key 'retryDelay' inside the detail object
                                # The structure provided in your log was a dict with specific keys.
                                if isinstance(detail, dict) and 'retryDelay' in detail:
                                    delay_str = detail['retryDelay']
                                    if delay_str.endswith('s'):
                                        wait_time = float(delay_str[:-1]) + 1.1
                                        break
                            if not details:
                                    logger.warning(f"Could not parse retryDelay from error: {e.details}. Using default backoff.")
                    except (ValueError, AttributeError) as parse_err:
                        logger.warning(f"Could not parse retryDelay from error: {parse_err}. Using default backoff.")

                    logger.warning(
                        f"Rate Limit Hit ({e.code}). Server requested wait: {wait_time:.4f}s. "
                        f"Retrying attempt {retries}/{max_retries}..."
                    )

                    time.sleep(wait_time)
        return wrapper
    return decorator

# --- 1. Taxonomy Management ---
def load_taxonomy(library_root: str) -> Dict[str, str]:
    """
    Builds the taxonomy map with two distinct passes:
    1. Collection: Resolves paths, normalizes keys, and loads raw descriptions.
    2. Inheritance: Propagates parent descriptions to children.
    """
    taxonomy_files = sorted(glob.glob(f"{library_root}/**/{TAXONOMY_FILENAME}", recursive=True))

    # Pass 1: Collect Raw Descriptions
    raw_map: Dict[str, str] = {}
    abs_lib_root = os.path.abspath(library_root)

    for file_path in taxonomy_files:
        try:
            # Determine directory relative to library root
            abs_file_dir = os.path.dirname(os.path.abspath(file_path))
            rel_dir = os.path.relpath(abs_file_dir, abs_lib_root)

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue

                    parts = line.split(' - ', 1)
                    if len(parts) != 2: continue

                    sub_path, description = parts[0].strip(), parts[1].strip()

                    # Construct the raw path
                    if rel_dir == ".":
                        # We are in the root directory
                        raw_key = sub_path
                    else:
                        # We are in a subdirectory
                        raw_key = os.path.join(rel_dir, sub_path)

                    # Normalize path to remove './', '..', and duplicate slashes
                    # This turns './Math' -> 'Math' and '.' -> '.'
                    norm_key = os.path.normpath(raw_key)

                    # Convert to forward slashes for internal consistency
                    key = norm_key.replace(os.sep, '/')

                    raw_map[key] = description

        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")

    # Pass 2: Propagate Descriptions (Inheritance)
    final_taxonomy = {}

    for path in sorted(raw_map.keys()):
        parts = path.split('/')
        description_chain = []

        # 1. Inherit from Root ('.') if defined and we are not currently '.'
        if '.' in raw_map and path != '.':
            description_chain.append(raw_map['.'])

        # 2. Inherit from Path Ancestors
        # Range logic:
        # For 'Math' (len 1): range(1, 1) -> Empty loop. Correct.
        # For 'Math/Algebra' (len 2): range(1, 2) -> i=1. parent='Math'. Correct.
        for i in range(1, len(parts)):
            parent_key = "/".join(parts[:i])
            if parent_key in raw_map:
                description_chain.append(raw_map[parent_key])

        # 3. Add Own Description
        description_chain.append(raw_map[path])

        final_taxonomy[path] = " - ".join(description_chain)

    return final_taxonomy

def format_taxonomy_prompt(taxonomy: Dict[str, str]) -> str:
    """Formats the taxonomy dictionary for the LLM prompt."""
    return "\n".join([f"- {path}: {desc}" for path, desc in sorted(taxonomy.items())])

def format_classification_example(taxonomy: Dict[str, str], max_samples: int = 10) -> str:
    """
    Generates a deterministic but diverse JSON example string from the taxonomy,
    interleaved with nulls to teach the LLM how to handle unclassifiable files.
    """
    keys = sorted(list(taxonomy.keys()))
    total_keys = len(keys)

    if total_keys == 0:
        return "[\"Finance/trading\", null, \"Computer science/algorithms\"]"

    # Reserve slots for 'null' to ensure we don't exceed max_samples
    # We want at least 2 nulls if possible, so we pick (max - 2) real categories
    num_real = max(1, min(total_keys, max_samples - 2))

    # Select keys deterministically using a stride to get a spread across the alphabet
    # e.g., if we have 100 keys and need 8, we take index 0, 12, 24...
    stride = total_keys // num_real
    samples = [keys[i * stride] for i in range(num_real)]

    # Insert 'None' (which json.dumps converts to 'null') at fixed positions
    # We place them in the middle so the model learns nulls can appear anywhere
    if len(samples) > 1:
        samples.insert(1, None)
    if len(samples) > 4:
        samples.insert(4, None)

    # Serialize to JSON string
    return json.dumps(samples)

# --- 2. Content Extraction ---
class FileHandler:
    """Handles content extraction for various file types."""

    _CID_PATTERN = re.compile(r'\(cid:\d+\)')
    _WHITESPACE_PATTERN = re.compile(r'\s+')

    @staticmethod
    def clean_text(text: str) -> str:
        if not text: return ""
        # 1. Remove PDF specific artifacts (CID codes)
        text = FileHandler._CID_PATTERN.sub('', text)

        # 2. Normalize Unicode (NFKC form)
        # This fixes separate accents, standardizes dashes, and fixes compatibility chars.
        text = unicodedata.normalize('NFKC', text)

        # 3. Filter Control Characters
        # We keep printable characters, newlines, and tabs.
        # We strip "C" category (Control, Format, Surrogate, Private Use)
        # except for standard whitespace.
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")

        # 4. Collapse Whitespace
        # Replaces multiple spaces/newlines with a single space to save LLM tokens.
        # If structure matters, you might want to preserve single \n, but for
        # classification, density is better.
        text = FileHandler._WHITESPACE_PATTERN.sub(' ', text).strip()
        return text

    @staticmethod
    def _extract_pdf(file_path: str) -> Optional[str]:
        try:
            reader = PdfReader(file_path)
            text_content = []
            for i, page in enumerate(reader.pages):
                if i > 1: break # Limit to first 2 pages
                extracted = page.extract_text()
                if extracted: text_content.append(extracted)
            return "\n".join(text_content)
        except Exception as e:
            logger.warning(f"PDF extract error {os.path.basename(file_path)}: {e}")
            return "[No content extracted]"

    @staticmethod
    def _extract_plain_text(file_path: str) -> Optional[str]:
        try:
            # Strict UTF-8 as per Pete's standards.
            # If standard text files are not UTF-8, they are likely legacy/garbage.
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(CONTENT_PREVIEW_LENGTH * 2) # Read a bit extra, truncate later
        except Exception as e:
            logger.warning(f"Text read error {os.path.basename(file_path)}: {e}")
            return "[No content extracted]"

    @staticmethod
    def get_content(file_path: str) -> str:
        """
        Returns a clean string snippet for the file.
        Returns explicit placeholder string if extraction is skipped/failed.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        raw_text = None

        if ext in SUPPORTED_EXTS_NO_CONTENT:
            return "[Content extraction skipped]"

        elif ext in SUPPORTED_EXTS_TEXT:
            raw_text = FileHandler._extract_plain_text(file_path)

        elif ext in SUPPORTED_EXTS_PDF:
            raw_text = FileHandler._extract_pdf(file_path)

        # Post-process
        if not raw_text:
            return "[No content extracted]"

        cleaned = FileHandler.clean_text(raw_text)

        # PDF Garbage check (only really needed for PDFs, but harmless for others)
        if ext in SUPPORTED_EXTS_PDF and (len(cleaned) < 50 or FileHandler._is_garbage(cleaned)):
             return "[Content unreadable/garbage]"

        return cleaned[:CONTENT_PREVIEW_LENGTH]

    @staticmethod
    def _is_garbage(text: str) -> bool:
        if len(text) < 50: return False
        alphanumeric = sum(c.isalnum() for c in text)
        return (alphanumeric / len(text)) < 0.5


def parse_response(response: str, expected_count: int, taxonomy: Dict) -> List[Optional[str]]:
    try:
        # Cleanup Markdown code blocks if present
        clean_resp = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_resp)

        if not isinstance(data, list) or len(data) != expected_count:
            logger.error(f"Invalid JSON structure. Got {len(data) if isinstance(data, list) else 'type mismatch'}, expected {expected_count}")
            return [None] * expected_count

        validated = []
        for item in data:
            if item in taxonomy:
                validated.append(item)
            else:
                if item: logger.warning(f"Invalid category returned: {item}")
                validated.append(None)
        return validated

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response: {response}")
        return [None] * expected_count

# --- 4. File Operations ---

def move_file(file_path: str, category: Optional[str], library_root: str):
    file_name = os.path.basename(file_path)
    if not category:
        logger.info(f"Unclassified: {file_path}")
        return

    dest_dir = os.path.join(library_root, *category.split('/'))
    dest_path = os.path.join(dest_dir, file_name)

    # usage of abspath ensures we compare the real filesystem paths
    if os.path.abspath(file_path) == os.path.abspath(dest_path):
        logger.info(f"Skipping move: {file_name} is already in {category}")
        return

    try:
        os.makedirs(dest_dir, exist_ok=True)

        # Check for name collision with a *different* file at destination
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(file_name)
            # Add timestamp to ensure uniqueness if multiple duplicates exist
            dest_path = os.path.join(dest_dir, f"{base}_{int(time.time())}_duplicate{ext}")

        shutil.move(file_path, dest_path)
        logger.info(f"Moved: {file_name} -> {category}")

    except OSError as e:
        logger.error(f"Error moving {file_name}: {e}")


def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="LLM-Powered PDF Classifier")
    parser.add_argument("input_path", type=str, help="Directory containing PDFs to classify")
    parser.add_argument("library_path", type=str, help="Root directory of the library/taxonomy")
    parser.add_argument("--api-key", type=str, default=os.environ.get("GEMINI_API_KEY"),
                        help="Google Gemini API Key (or set GEMINI_API_KEY env var)")
    parser.add_argument("-r", "--recursive", action="store_true",
                            help="Recursively process subdirectories")

    args = parser.parse_args()

    if not args.api_key:
        logger.critical("API Key is missing. Set GEMINI_API_KEY or pass --api-key.")
        sys.exit(1)

    if not os.path.isdir(args.input_path) or not os.path.isdir(args.library_path):
        logger.critical("Invalid input or library directories.")
        sys.exit(1)

    # 2. Initialization
    logger.info("Initializing Taxonomy...")
    taxonomy_map = load_taxonomy(args.library_path)
    if not taxonomy_map:
        logger.error("Taxonomy empty. Check taxonomy.md files.")
        sys.exit(1)

    logger.info(f"Loaded {len(taxonomy_map)} categories.")

    # Discovery
    files_to_process = []
    logger.info("Scanning files...")

    # Discovery Logic (Recursive vs Flat)
    if args.recursive:
        # Recursive: Walk the tree
        logger.info(f"Scanning {args.input_path} recursively...")
        for root, _, files in os.walk(args.input_path, followlinks=False):
            for f in files:
                if os.path.splitext(f)[1].lower() in ALL_SUPPORTED_EXTS:
                    files_to_process.append(os.path.join(root, f))
    else:
        logger.info(f"Scanning {args.input_path} (non-recursive)...")
        try:
            with os.scandir(args.input_path) as entries:
                for e in entries:
                    if e.is_file() and os.path.splitext(e.name)[1].lower() in ALL_SUPPORTED_EXTS:
                        files_to_process.append(e.path)
        except OSError as e:
            logger.error(f"Scan error: {e}")

    if not files_to_process:
        logger.info("No matching files found.")
        return

    files_to_process.sort()

    #print(taxonomy_map)
    #print(format_classification_example(taxonomy_map))
    #exit(0)
    # Construct Prompt
    # Ensure taxonomy_str is pre-computed to keep f-string clean
    taxonomy_str = "\n".join([f"- {k}: {v}" for k,v in sorted(taxonomy_map.items())])
    examples_str = format_classification_example(taxonomy_map)
    system_prompt = (
            "You are a strict librarian. Classify the provided documents into the Taxonomy below based on their Name, Snippet, and Location.\n"
            "Output strictly a JSON list of strings (or null) corresponding exactly to the order of input files.\n\n"
            "RULES:\n"
            "1. STRICT TAXONOMY: Use category paths EXACTLY as listed. Do not invent categories.\n"
            "2. CONSERVATIVE ASSIGNMENT: Do not force a match based on partial keywords. If a file is a 'Physics Textbook' but you only have 'Math/Textbooks', return null. The specific domain must match.\n"
            "3. HIGH CONFIDENCE: If a file does not fit clearly, return null.\n\n"
            f"TAXONOMY:\n{taxonomy_str}\n\n"
            f"EXAMPLE OUTPUT:\n{examples_str}"
        )

    # print(system_prompt)
    # exit(0)

    # Define the schema: List[Optional[str]]
    # strict=True ensures the model adheres exactly to this structure
    response_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.STRING,
            nullable=True
        )
    )

    client = genai.Client(api_key=args.api_key)
    llm_cache = None
    # Try to enable cache
    try:
        # Create a cached content object
        llm_cache = client.caches.create(
            model=LLM_MODEL,
            config=types.CreateCachedContentConfig(
              system_instruction=system_prompt,
            )
        )
        # Display the cache details
        logger.info(f'Cache configured: {llm_cache}')
    except Exception as e:
        logger.error(f'Cache not configured: {str(e)}')

    @retry_on_429()
    def classify(prompt: str):
        """
        Wraps the API call with the retry logic.
        """
        if llm_cache:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    cached_content=llm_cache.name,
                    temperature=0.0, # Deterministic output
                    response_mime_type="application/json",
                    response_schema = response_schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
        else:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0, # Deterministic output
                    response_mime_type="application/json",
                    response_schema = response_schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            )
        print(response.text)
        return response

    logger.info(f"Processing {len(files_to_process)} file(s)...")

    try:
        # 4. Processing Loop in batches
        for i in range(0, len(files_to_process), BATCH_SIZE):
            batch = files_to_process[i : i + BATCH_SIZE]

            prompt=""
            for idx, fpath in enumerate(batch):
                content = FileHandler.get_content(fpath)
                rel_location = os.path.dirname(os.path.relpath(fpath, args.input_path))
                prompt += (
                                f"--- FILE {idx + 1} ---\n"
                                f"Name: {os.path.basename(fpath)}\n"
                                f"Snippet: {content}\n"
                                f"Location: {rel_location}\n"
                )
                #prompt += f"--- FILE {i+1} ---\nName: {os.path.basename(fpath)}\nSnippet: {content}\nLocation: {os.path.dirname(fpath)}\n"
                # print(prompt)
                # exit(0)
            try:
                resp = classify(prompt)
            except Exception as e:
                logger.error(f"Gemini API Error: {e}")
                continue
            categories = parse_response(resp.text, len(batch), taxonomy_map)

            # Move Files
            for fpath, cat in zip(batch, categories):
                move_file(fpath, cat, args.library_path)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unexpected fatal error: {e}")
        if llm_cache:
                client.caches.delete(name = llm_cache.name)
        sys.exit(1)

    if llm_cache:
        client.caches.delete(name = llm_cache.name)
    logger.info("Classification complete.")

if __name__ == "__main__":
    main()
