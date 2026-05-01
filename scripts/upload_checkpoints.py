"""Upload all DiffusionDPO LoRA checkpoints from a folder to Hugging Face Hub.

Finds LoRA checkpoint directories (checkpoint-*) inside the given folder
and uploads each one to a HF repo.

Usage:
    # Upload all LoRA checkpoints as subdirectories in one repo (default)
    python scripts/upload_checkpoints.py \
        --checkpoints_dir /path/to/output_dir \
        --repo_id username/my-diffusiondpo-lora

    # Upload each LoRA checkpoint to a separate branch
    python scripts/upload_checkpoints.py \
        --checkpoints_dir /path/to/output_dir \
        --repo_id username/my-diffusiondpo-lora \
        --separate_branches

    # Skip training state files to save space
    python scripts/upload_checkpoints.py \
        --checkpoints_dir /path/to/output_dir \
        --repo_id username/my-diffusiondpo-lora \
        --skip_training_state
"""

import argparse
import glob
import os
import re
import sys

from huggingface_hub import HfApi, create_repo


CHECKPOINT_PATTERN = "checkpoint-*"

TRAINING_STATE_FILES = {
    "training_state.pt",
}


def find_checkpoints(checkpoints_dir, pattern):
    """Find all LoRA checkpoint directories matching the pattern."""
    found = []
    for m in glob.glob(os.path.join(checkpoints_dir, pattern)):
        if os.path.isdir(m):
            found.append(m)
    return sorted(found, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path):
    """Sort checkpoints by step number."""
    name = os.path.basename(path)
    numbers = re.findall(r"\d+", name)
    return [int(n) for n in numbers] if numbers else [0]


def upload_checkpoints(
    checkpoints_dir,
    repo_id,
    pattern,
    separate_branches,
    skip_training_state,
    private,
    token,
):
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    checkpoint_dirs = find_checkpoints(checkpoints_dir, pattern)
    if not checkpoint_dirs:
        print(f"No LoRA checkpoint directories found in {checkpoints_dir}")
        print(f"Looked for pattern: {pattern}")
        sys.exit(1)

    print(f"Found {len(checkpoint_dirs)} LoRA checkpoint(s):")
    for d in checkpoint_dirs:
        print(f"  {os.path.basename(d)}")
    print()

    ignore_patterns = []
    if skip_training_state:
        ignore_patterns = list(TRAINING_STATE_FILES)
        print(f"Skipping training state files: {ignore_patterns}")
        print()

    for i, ckpt_dir in enumerate(checkpoint_dirs, 1):
        name = os.path.basename(ckpt_dir)
        print(f"[{i}/{len(checkpoint_dirs)}] Uploading {name} ...")

        if separate_branches:
            # Each checkpoint goes to its own branch at the repo root
            branch = name
            try:
                api.create_branch(repo_id, repo_type="model", branch=branch)
            except Exception:
                pass  # Branch may already exist

            api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message=f"Upload {name}",
                ignore_patterns=ignore_patterns or None,
            )
            print(f"  -> uploaded to branch '{branch}'")
        else:
            # All checkpoints go as subdirectories on main branch
            api.upload_folder(
                folder_path=ckpt_dir,
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=name,
                commit_message=f"Upload {name}",
                ignore_patterns=ignore_patterns or None,
            )
            print(f"  -> uploaded to {repo_id}/{name}")

    print()
    print(f"Done. All LoRA checkpoints uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload DiffusionDPO LoRA checkpoints to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Path to the folder containing checkpoint-* directories",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. username/my-diffusiondpo-lora)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=CHECKPOINT_PATTERN,
        help=f'Glob pattern to match checkpoint dirs (default: "{CHECKPOINT_PATTERN}")',
    )
    parser.add_argument(
        "--separate_branches",
        action="store_true",
        help="Upload each LoRA checkpoint to a separate branch instead of as subdirectories",
    )
    parser.add_argument(
        "--skip_training_state",
        action="store_true",
        help="Skip training_state.pt files to save space",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token. If not set, uses the cached token from `huggingface-cli login`.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoints_dir):
        print(f"Error: {args.checkpoints_dir} is not a directory")
        sys.exit(1)

    upload_checkpoints(
        checkpoints_dir=args.checkpoints_dir,
        repo_id=args.repo_id,
        pattern=args.pattern,
        separate_branches=args.separate_branches,
        skip_training_state=args.skip_training_state,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
