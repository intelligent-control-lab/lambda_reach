from glob import glob

from setuptools import find_packages, setup

package_name = "mocap_avoid_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shangtao",
    maintainer_email="shangtao@example.com",
    description="Bridge MoCap rigid-body poses into collision-avoidance observations.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mocap_avoid_bridge_node = mocap_avoid_bridge.mocap_avoid_bridge_node:main",
        ],
    },
)
