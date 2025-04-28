"""
IEMOCAP Dataset Structure Explorer

This script explores and displays the directory structure of the IEMOCAP dataset
in a manageable way, showing only a few sample files from directories with many files.
"""

import argparse
import json
import random
from pathlib import Path


def explore_directory(directory, max_files=5, max_depth=None, current_depth=0):
    """
    Recursively explore a directory and collect its structure.

    Args:
        directory: Directory path to explore
        max_files: Maximum number of sample files to include per directory
        max_depth: Maximum depth to explore (None for unlimited)
        current_depth: Current exploration depth

    Returns:
        Dictionary representing the directory structure
    """
    if max_depth is not None and current_depth > max_depth:
        return "..."

    path = Path(directory)

    if not path.exists():
        return "ERROR: Path does not exist"

    if not path.is_dir():
        return f"FILE: {path.name}"

    result = {}
    subdirs = []
    files = []

    # Collect all items in the directory
    for item in path.iterdir():
        if item.name.startswith((".", "_", "~")):
            continue

        if item.is_dir():
            subdirs.append(item)
        else:
            files.append(item)

    # Process subdirectories (depth-first)
    for subdir in sorted(subdirs):
        result[f"DIR: {subdir.name}/"] = explore_directory(subdir, max_files, max_depth, current_depth + 1)

    # Process files
    if files:
        if len(files) > max_files:
            # Sample a few files if there are many
            sampled_files = random.sample(files, max_files)
            file_names = [f.name for f in sorted(sampled_files)]
            result["FILES"] = f"{len(files)} files, examples: {file_names}"
        else:
            # List all files if there are few
            result["FILES"] = [f.name for f in sorted(files)]

    return result


def find_label_files(root_dir):
    """Find and read a sample of label files to understand the format"""
    label_files = []

    # Look for label files in common locations
    patterns = [
        "**/EmoEvaluation/**/*.txt",
        "**/Categorical/**/*.txt",
        "**/dialog/EmoEvaluation/**/*.txt",
    ]

    for pattern in patterns:
        found = list(Path(root_dir).glob(pattern))
        if found:
            label_files.extend(found)

    # Sample up to 3 files
    if label_files:
        sample_files = random.sample(label_files, min(3, len(label_files)))

        print("\n=== Sample Label File Content ===")
        for file in sample_files:
            print(f"\nFile: {file}")
            try:
                with open(file, "r") as f:
                    # Read up to 10 lines
                    lines = [next(f).strip() for _ in range(10) if f]
                    print("First 10 lines:")
                    for line in lines:
                        print(f"  {line}")
            except Exception as e:
                print(f"Error reading file: {e}")


def find_audio_files(root_dir):
    """Find and check sample audio files"""
    audio_files = []

    # Look for audio files in common locations
    patterns = [
        "**/sentences/wav/**/*.wav",
        "**/wav/**/*.wav",
    ]

    for pattern in patterns:
        found = list(Path(root_dir).glob(pattern))
        if found:
            audio_files.extend(found)

    # Sample up to 5 files
    if audio_files:
        sample_files = random.sample(audio_files, min(5, len(audio_files)))

        print("\n=== Sample Audio File Paths ===")
        for file in sample_files:
            print(f"Audio file: {file}")
            print(f"Exists: {file.exists()}")
            print(f"Size: {file.stat().st_size} bytes")

            # Check if there's a corresponding label file
            dialog_id = file.parent.name
            potential_label_files = list(Path(root_dir).glob(f"**/EmoEvaluation/**/{dialog_id}*.txt"))
            if potential_label_files:
                print(f"Potential label file: {potential_label_files[0]}")


def find_video_files(root_dir):
    """Find and check sample video files"""
    video_files = []

    # Look for video files in common locations
    patterns = [
        "**/dialog/avi/**/*.avi",
        "**/dialog/video/**/*.mp4",
        "**/dialog/video/**/*.avi",
    ]

    for pattern in patterns:
        found = list(Path(root_dir).glob(pattern))
        if found:
            video_files.extend(found)

    # Sample up to 3 files
    if video_files:
        sample_files = random.sample(video_files, min(3, len(video_files)))

        print("\n=== Sample Video File Paths ===")
        for file in sample_files:
            print(f"Video file: {file}")
            print(f"Exists: {file.exists()}")
            print(f"Size: {file.stat().st_size} bytes")


def pretty_print_dict(d, indent=0):
    """Pretty print a dictionary with proper indentation"""
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + str(key))
            pretty_print_dict(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Explore IEMOCAP dataset structure")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the IEMOCAP dataset directory")
    parser.add_argument("--max_files", type=int, default=5, help="Maximum number of sample files to display per directory")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum directory depth to explore")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file to save the structure")

    args = parser.parse_args()

    print(f"Exploring IEMOCAP dataset at: {args.dataset_dir}")

    # Explore directory structure
    structure = explore_directory(args.dataset_dir, args.max_files, args.max_depth)

    # Print the structure in a readable format
    print("\n=== IEMOCAP Directory Structure ===")
    pretty_print_dict(structure)

    # Find and examine label files
    find_label_files(args.dataset_dir)

    # Find and examine audio files
    find_audio_files(args.dataset_dir)

    # Find and examine video files
    find_video_files(args.dataset_dir)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(structure, f, indent=2)
        print(f"\nStructure saved to {args.output}")


if __name__ == "__main__":
    main()
