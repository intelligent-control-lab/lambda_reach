from setuptools import find_packages, setup

subpackage_names = find_packages(where=".")
packages = ["safety_value"] + [f"safety_value.{name}" for name in subpackage_names]

setup(
    name="safety-value",
    version="0.1.0",
    description="Safety value learning tools for the HJ humanoid project",
    license="PolyForm-Noncommercial-1.0.0",
    packages=packages,
    package_dir={"safety_value": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "h5py>=3.0.0",
        "tqdm>=4.60.0",
    ],
)
