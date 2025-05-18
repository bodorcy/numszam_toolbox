from setuptools import setup, find_packages

setup(
    name="numszam_toolbox",
    version="0.1.0",
    description="Egy fuggvenykonyvtar, es egyszeru webapp a Numerikus Számítások tárgyhoz.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Vincze Nándor",
    author_email="nvincze@inf.u-szeged.hu",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "flask",
        "matplotlib",
        "numpy",
        "pytest",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/bodorcy/numszam_toolbox",
)
