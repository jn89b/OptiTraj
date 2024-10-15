from setuptools import setup, find_packages

setup(
    name="OptiTraj",  # Package name
    version="0.1.0",  # Initial version
    author="Justin Nguyen",  # Replace with your name
    author_email="jnguyenblue2804@gmail.com",  # Replace with your email
    # Short description
    description="A Python package for Model Predictive Control (MPC), Trajectory Optimization, and Path Planning",
    long_description=open('README.md').read(),  # Include a README if available
    long_description_content_type="text/markdown",
    url="https://github.com/jn89b/OptiTraj/tree/master/optitraj",  # Replace with your repo
    packages=find_packages(),  # Automatically find the package in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",  # Add dependencies like numpy, scipy, etc.
        "scipy",
        "casadi",
        "matplotlib",
    ],
)
