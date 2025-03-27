from setuptools import setup, find_packages

setup(
    name="wrmodes",
    version="0.1.0",
    description="A tool to calculate cutoff frequencies and propagating modes in rectangular waveguides.",
    author="Daniel Hachmeister",  # Replace with your name
    author_email="daniel.hachmeister@tecnico.ulisboa.pt",  # Replace with your email
    packages=find_packages(),
    install_requires=[
        "typer[all]",  # For CLI functionality
        "scipy",       # For scientific constants
    ],
    entry_points={
        "console_scripts": [
            "waveguide-modes=wrmodes.main:script",  # Expose the CLI
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
