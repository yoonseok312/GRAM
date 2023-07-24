from setuptools import (
    setup,
    find_packages,
)

setup(
    name="examples",
    version="0.1.1",
    description="",
    long_description="",
    packages=find_packages(include=["examples", "examples.*"]),
    package_data={},
    zip_safe=False,
)
