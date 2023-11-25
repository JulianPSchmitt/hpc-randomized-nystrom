# Setup code as package to access files from subfolders
# Done using "pip install -e ."

from setuptools import setup, find_packages

_requirements = ["numpy", "mpi4py", "matplotlib", "scipy", "tqdm"]

setup(
    name="randnystrom",
    version="0.1",
    author="Christian Mikkelstrup and Julian Schmitt",
    author_email="christian.mikkelstrup@epfl.ch, julian.schmitt@epfl.ch",
    description="Usage and study of the randomized Nyström algorithm.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="code"),
    package_dir={"": "code"},
    include_package_data=True,
    setup_requires=["setuptools"],
    install_requires=_requirements,
    license="MIT",
)
