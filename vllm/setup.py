from setuptools import find_packages, setup

setup(
    name="vllm-ac",
    version="0.3.0",
    packages=find_packages(),
    package_data={"vllm_ac": ["ARITHMETIC_CODING_LICENSE.txt"]},
    python_requires=">=3.10",
    install_requires=["vllm==0.11.0", "transformers==4.55.2"],
    entry_points={
        "vllm.general_plugins": ["teacher_forced_scoring = vllm_ac:register"]
    },
)
