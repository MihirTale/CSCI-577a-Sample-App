#!/usr/bin/env python3
"""
Model Serving Pipeline - Step 3
Simulates a missing environment variable / secret that is required at runtime.

This is one of the most common real-life CI/CD failures: a secret or config
value exists locally (or in a previous environment) but was never added to the
CI secrets store, causing a hard crash at the point of first use.
"""

import os
import sys
import traceback


REQUIRED_ENV_VARS = [
    "MODEL_SERVING_ENDPOINT",
    "MODEL_API_SECRET_KEY",
    "INFERENCE_TIMEOUT_MS",
    "FEATURE_STORE_DSN",
]


def validate_environment() -> dict:
    """
    Read all required environment variables and return them as a dict.

    Raises KeyError listing every missing variable so the developer sees
    all problems at once rather than fixing one and hitting the next.
    """
    config = {}
    missing = []

    for var in REQUIRED_ENV_VARS:
        value = os.environ.get(var)
        if value is None:
            missing.append(var)
        else:
            config[var] = value

    if missing:
        raise KeyError(
            f"Required environment variable(s) not set: {missing}\n"
            "  These must be configured as repository secrets or environment variables.\n"
            "  Check Settings → Secrets and variables → Actions in your GitHub repository.\n"
            "  Local .env files are NOT automatically loaded inside GitHub Actions runners."
        )

    return config


def connect_model_endpoint(config: dict) -> None:
    """Simulate connecting to the model serving endpoint."""
    endpoint = config["MODEL_SERVING_ENDPOINT"]
    print(f"[INFO] Connecting to model endpoint: {endpoint}")
    print("[INFO] Authenticating with MODEL_API_SECRET_KEY...")
    print(f"[INFO] Timeout configured: {config['INFERENCE_TIMEOUT_MS']} ms")
    print(f"[INFO] Feature store DSN  : {config['FEATURE_STORE_DSN']}")
    print("[INFO] Connection established.")


if __name__ == "__main__":
    print("[INFO] ============================================================")
    print("[INFO]  Step 3: Model Serving — Environment Validation")
    print("[INFO] ============================================================")
    print("[INFO] Script  : model_server.py")
    print("[INFO] Purpose : Connect to model serving endpoint and validate runtime config")
    print()

    try:
        config = validate_environment()
        connect_model_endpoint(config)
        print("[INFO] Step 3 PASSED.")
    except KeyError as e:
        print()
        print("[ERROR] *** MISSING ENVIRONMENT VARIABLE(S) ***", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "[ERROR] Step 3 FAILED: one or more required environment variables are not set.",
            file=sys.stderr,
        )
        print("[ERROR] Traceback (most recent call last):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
