#!/usr/bin/env python3
"""
Dependency Resolution Pipeline - Step 4
Simulates a dependency/version conflict that causes an ImportError at runtime.

In real CI/CD this surfaces when:
  - requirements.txt pins conflicting transitive dependencies
  - A library upgrade introduces a breaking change in its public API
  - Two packages require mutually exclusive versions of a shared dependency
  - A package is installed but its optional extras are missing
"""

import sys
import traceback


# ---------------------------------------------------------------------------
# Simulate the import chain that fails due to the conflict
# ---------------------------------------------------------------------------

class _FakeConflict:
    """
    Mimics the error that numpy/pandas/sklearn raise when their C-extension
    ABI is compiled against an incompatible runtime.
    """

    def __init__(self, name: str, installed: str, required: str, needed_by: str):
        self.name = name
        self.installed = installed
        self.required = required
        self.needed_by = needed_by

    def trigger(self):
        raise ImportError(
            f"cannot import name 'DistributionNotFound' from '{self.name}' "
            f"({self.name}-{self.installed})\n\n"
            f"  Installed : {self.name}=={self.installed}\n"
            f"  Required  : {self.name}{self.required}  "
            f"  (needed by {self.needed_by})\n\n"
            f"  This usually means two packages in requirements.txt have\n"
            f"  conflicting constraints on {self.name}.\n\n"
            f"  Reproduction steps:\n"
            f"    pip install scikit-learn==1.3.2\n"
            f"    pip install imbalanced-learn==0.11.0\n"
            f"  Both claim to need scipy, but with incompatible version bounds:\n"
            f"    scikit-learn 1.3.2  -> scipy>=1.5.0,<2.0.0\n"
            f"    imbalanced-learn 0.11.0 -> scipy>=1.5.0,<1.12.0\n"
            f"  The resolver installed scipy==2.1.3, satisfying scikit-learn\n"
            f"  but breaking imbalanced-learn's upper bound.\n\n"
            f"  Fix: pin scipy to a version compatible with both:\n"
            f"    scipy>=1.5.0,<1.12.0\n"
            f"  or upgrade imbalanced-learn to 0.12.x which lifted the cap."
        )


def import_ml_pipeline_dependencies() -> None:
    """
    Import the ML pipeline dependencies.  Dies if any version constraint
    is violated.
    """
    print("[INFO] Importing numpy          ... OK  (numpy==1.26.4)")
    print("[INFO] Importing pandas         ... OK  (pandas==2.2.1)")
    print("[INFO] Importing scikit-learn   ... OK  (scikit-learn==1.3.2)")
    print("[INFO] Importing scipy          ... OK  (scipy==2.1.3)")
    print("[INFO] Importing imbalanced-learn ...")

    conflict = _FakeConflict(
        name="scipy",
        installed="2.1.3",
        required=">=1.5.0,<1.12.0",
        needed_by="imbalanced-learn==0.11.0",
    )
    conflict.trigger()


def run_pipeline() -> None:
    import_ml_pipeline_dependencies()
    print("[INFO] All dependencies loaded.")
    print("[INFO] Running SMOTE oversampling on imbalanced training set...")
    print("[INFO] Step 4 PASSED.")


if __name__ == "__main__":
    print("[INFO] ============================================================")
    print("[INFO]  Step 4: ML Pipeline — Dependency Validation")
    print("[INFO] ============================================================")
    print("[INFO] Script  : ml_trainer.py")
    print("[INFO] Purpose : Import ML pipeline dependencies before model training")
    print()

    try:
        run_pipeline()
    except ImportError as e:
        print()
        print("[ERROR] *** DEPENDENCY / VERSION CONFLICT ***", file=sys.stderr)
        print(f"[ERROR] {e}", file=sys.stderr)
        print(
            "[ERROR] Step 4 FAILED: package import failed due to conflicting version constraints.",
            file=sys.stderr,
        )
        print("[ERROR] Traceback (most recent call last):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
