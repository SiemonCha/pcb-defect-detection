"""
Quick analysis suite orchestration.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, Optional

from analysis.dataset import main as dataset_analysis_main
from analysis.failures import main as failures_analysis_main
from evaluation.interpretability import (
    main as interpretability_analysis_main,
)
from evaluation.robustness import main as robustness_main


def find_best_model() -> Optional[str]:
    """Locate the newest trained model."""
    patterns = [
        "runs/train/production_yolov8s*/weights/best.pt",
        "runs/train/baseline_yolov8n*/weights/best.pt",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    return None


def run_callable(func, description: str, args: Optional[Iterable[str]] = None) -> bool:
    """Execute a callable that implements a CLI-like main()."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")

    try:
        if args is None:
            result = func()
        else:
            result = func(args=args)

        if result is None or result is True or result == 0:
            print(f"\n{description} completed")
            return True

        print(f"\n{description} failed")
        return False
    except SystemExit as exc:
        if exc.code == 0:
            print(f"\n{description} completed")
            return True
        print(f"\n{description} failed with exit code {exc.code}")
        return False
    except Exception as error:
        print(f"\n{description} error: {error}")
        return False


def run_quick_analysis(
    model_path: Optional[str] = None,
    skip_failures: bool = False,
    skip_robustness: bool = False,
    failure_top: int = 20,
    interpretability_samples: int = 10,
    robustness_samples: int = 10,
) -> bool:
    """Run the quick analysis suite."""
    print(f"\n{'='*60}")
    print("QUICK ANALYSIS SUITE")
    print(f"{'='*60}")

    model_path = model_path or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print("\nxxxx No trained model found")
        print("   Train a model first:")
        print("   python -m cli train-baseline")
        return False

    print(f"Model: {model_path}")
    results = {}

    # Dataset analysis
    print(f"\n{'='*60}")
    print("PHASE 1/4: Dataset Analysis")
    print(f"{'='*60}")
    results["dataset"] = run_callable(dataset_analysis_main, "Dataset Analysis")

    # Failure analysis
    if skip_failures:
        print("\nSkipping failure analysis")
        results["failures"] = None
    else:
        print(f"\n{'='*60}")
        print("PHASE 2/4: Failure Case Analysis")
        print(f"{'='*60}")
        args = ["--model", model_path, "--top", str(failure_top)]
        results["failures"] = run_callable(failures_analysis_main, "Failure Analysis", args=args)

    # Interpretability
    print(f"\n{'='*60}")
    print("PHASE 3/4: Model Interpretability")
    print(f"{'='*60}")
    args = ["--model", model_path, "--samples", str(interpretability_samples)]
    results["interpretability"] = run_callable(
        interpretability_analysis_main,
        "Interpretability Analysis",
        args=args,
    )

    # Robustness
    if skip_robustness:
        print("\nSkipping robustness testing")
        results["robustness"] = None
    else:
        print(f"\n{'='*60}")
        print("PHASE 4/4: Robustness Testing")
        print(f"{'='*60}")
        args = ["--model", model_path, "--samples", str(robustness_samples)]
        results["robustness"] = run_callable(
            robustness_main,
            "Robustness Testing",
            args=args,
        )

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")

    completed = [name for name, success in results.items() if success is True]
    failed = [name for name, success in results.items() if success is False]
    skipped = [name for name, success in results.items() if success is None]

    total_run = len([success for success in results.values() if success is not None])
    print(f"\nCompleted: {len(completed)}/{total_run}")

    if completed:
        print("\nCompleted:")
        for task in completed:
            print(f"   - {task}")

    if failed:
        print("\nFailed:")
        for task in failed:
            print(f"   - {task}")

    if skipped:
        print("\nSkipped:")
        for task in skipped:
            print(f"   - {task}")

    print(f"\n{'='*60}")
    print("RESULTS LOCATION")
    print(f"{'='*60}")
    for directory in [
        "logs/dataset_analysis",
        "logs/failure_analysis",
        "logs/interpretability",
        "logs/robustness",
    ]:
        path = Path(directory)
        if path.exists():
            files = list(path.glob("*"))
            print(f"   {directory}/ ({len(files)} files)")

    print(f"\n{'='*60}")
    print("WHAT TO DO WITH RESULTS")
    print(f"{'='*60}")
    print("\nFor academic submission:")
    print("   1. Include class distribution chart from dataset_analysis/")
    print("   2. Show 5-10 failure cases from failure_analysis/")
    print("   3. Add 3-5 attention maps from interpretability/")
    print("   4. Report robustness scores from robustness/")

    print("\nFor presentations:")
    print("   - Dataset analysis: demonstrates understanding of the data")
    print("   - Failure cases: communicates limitations")
    print("   - Attention maps: highlights model focus")
    print("   - Robustness: illustrates performance under corruptions")

    print(f"\n{'='*60}")
    print("OPTIONAL NEXT STEPS")
    print(f"{'='*60}")
    print("\nFor deeper analysis (time-intensive):")
    print("   - Run training.cross_validation for statistical confidence intervals")
    print("   - Run training.hyperparameter to tune training parameters")

    print("\nFor production deployment:")
    print("   - Quantize the model for INT8 inference")
    print("   - Monitor live inference performance")

    success = len(failed) == 0
    print(f"\n{'='*60}")
    if success:
        print("All analyses completed successfully.")
    else:
        print("Some analyses failed. Review the errors above.")
    print(f"{'='*60}\n")

    return success


def main(args: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness testing")
    parser.add_argument("--skip-failures", action="store_true", help="Skip failure analysis")
    parser.add_argument("--failure-top", type=int, default=20, help="Number of failures to visualize")
    parser.add_argument("--interpretability-samples", type=int, default=10, help="Attention samples")
    parser.add_argument("--robustness-samples", type=int, default=10, help="Robustness samples")
    parsed = parser.parse_args(args=args)

    success = run_quick_analysis(
        model_path=parsed.model,
        skip_failures=parsed.skip_failures,
        skip_robustness=parsed.skip_robustness,
        failure_top=parsed.failure_top,
        interpretability_samples=parsed.interpretability_samples,
        robustness_samples=parsed.robustness_samples,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

