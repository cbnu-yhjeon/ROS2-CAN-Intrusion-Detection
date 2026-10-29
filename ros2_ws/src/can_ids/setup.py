from glob import glob

from setuptools import find_packages, setup

package_name = "can_ids"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/models", glob("models/*.json") + glob("models/*.onnx") + glob("models/*.ts")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yhjeon",
    maintainer_email="yhjeon@avgenius.kr",
    description="Real-time CAN bus intrusion detection node",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ids_node = can_ids.ids_node:main",
            "socketcan_bridge = can_ids.socketcan_bridge:main",
        ],
    },
)
