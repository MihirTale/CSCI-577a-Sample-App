#!/usr/bin/env python3
"""
Data Loading Pipeline - Step 1
Simulates an Out-of-Memory (OOM) error encountered during large dataset processing.
"""

import sys
import traceback


def load_dataset(filename: str) -> list:
    """
    Load a large dataset into memory for model training.

    In production this reads a multi-GB CSV file entirely into memory,
    which will exhaust available RAM on the runner.
    """
    print(f"[INFO] Opening file: {filename}")
    print("[INFO] Estimated file size: 52.4 GB")
    print("[INFO] Reading rows into memory buffer...")

    data = []
    batch_size = 50_000

    for batch_num in range(1, 10_000):
        print(f"[INFO] Loaded batch {batch_num} ({batch_size * batch_num:,} rows so far...)")

        # After a few batches, the process exceeds available memory
        if batch_num >= 4:
            raise MemoryError(
                f"Cannot allocate array of size {batch_size * batch_num * 64:,} bytes.\n"
                f"  Current process RSS: 6,442,450,944 bytes (6.0 GB)\n"
                f"  System available memory: 524,288 bytes (512 KB)\n"
                f"  Requested next allocation: {batch_size * 64:,} bytes\n"
                "Hint: use --batch-size or stream the file instead of loading it entirely."
            )

        data.extend([0.0] * batch_size)

    return data


if __name__ == "__main__":
    print("[INFO] ============================================================")
    print("[INFO]  Step 1: Load Training Dataset")
    print("[INFO] ============================================================")
    print("[INFO] Script      : data_loader.py")
    print("[INFO] Purpose     : Load full training data for downstream model training")
    print("[INFO] Source file : training_data_2024_v2.csv")
    print()

    try:
        dataset = load_dataset("training_data_2024_v2.csv")
        print(f"[INFO] Successfully loaded {len(dataset):,} rows.")
        print("[INFO] Step 1 PASSED.")
    except MemoryError as e:
        print()
        print("[ERROR] *** OUT-OF-MEMORY ERROR ***", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        print("[ERROR] Step 1 FAILED: process exhausted available memory.", file=sys.stderr)
        print("[ERROR] Traceback (most recent call last):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
