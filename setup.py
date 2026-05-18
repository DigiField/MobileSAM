# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="mobile_sam",
    version="1.0+f2",
    install_requires=["torchpack", "onnx", "onnxsim", "opencv-python"],
    packages=find_packages(exclude="notebooks"),
    package_data={'mobile_sam_v2': ['ultralytics/*/cfg/*.yaml', 'ultralytics/models/*/*.yaml']},
    include_package_data=True,
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
