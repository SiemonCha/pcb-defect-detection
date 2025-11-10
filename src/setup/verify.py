"""
Dependency verification utility.
"""

from __future__ import annotations

import argparse
from typing import Optional


def check_import(package_name: str, import_name: Optional[str] = None) -> bool:
    """Attempt to import a package and report its version."""
    target = import_name or package_name
    try:
        module = __import__(target)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK]     {package_name:<20s} (version: {version})")
        return True
    except ImportError:
        print(f"[MISSING] {package_name:<20s} - not installed")
        return False


def main(_: argparse.Namespace | None = None) -> int:
    """Verify core and optional dependencies."""
    print("=" * 60)
    print("DEPENDENCY VERIFICATION")
    print("=" * 60)

    print("\nCore Dependencies (existing):")
    core_checks = [
        check_import("torch"),
        check_import("ultralytics"),
        check_import("numpy"),
        check_import("matplotlib"),
        check_import("seaborn"),
        check_import("opencv-python", "cv2"),
        check_import("PIL"),
        check_import("yaml"),
    ]

    print("\nNew Dependencies:")
    new_checks = [
        check_import("optuna"),
        check_import("scipy"),
    ]
    core_checks.extend(new_checks)

    print("\nOptional Dependencies:")
    optional_checks = [
        check_import("onnx"),
        check_import("onnxruntime"),
        check_import("fastapi"),
        check_import("uvicorn"),
    ]

    print("\n" + "=" * 60)

    missing_core = sum(1 for result in core_checks if not result)
    missing_optional = sum(1 for result in optional_checks if not result)

    if all(core_checks):
        print("All core dependencies installed.")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python -m cli quick-analysis")
        print("  python -m training.cross_validation --quick")
        print("  python -m training.hyperparameter --quick")

        if missing_optional > 0:
            print(f"\nNote: {missing_optional} optional dependencies missing")
            print("These are only needed for specific features:")
            print("  - onnx/onnxruntime: For model quantization")
            print("  - fastapi/uvicorn: For API deployment")

        return 0

    print(f"{missing_core} core dependencies missing.")
    print("=" * 60)
    print("\nInstall missing packages:")

    missing = []
    names = [
        ("torch", "torch torchvision"),
        ("ultralytics", "ultralytics"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("opencv-python", "opencv-python"),
        ("PIL", "pillow"),
        ("yaml", "pyyaml"),
        ("optuna", "optuna"),
        ("scipy", "scipy"),
    ]
    for check, (pkg, install_name) in zip(core_checks, names):
        if not check:
            missing.append(install_name)

    if missing:
        print(f"  pip install {' '.join(missing)}")

    return 1


if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)

