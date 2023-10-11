from setuptools import setup, find_packages

PACKAGENAME = "diffsmhm"
VERSION = "0.0.dev"

setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Andrew Hearin, Matthew Becker, Joseph Wick",
    author_email="mrbecker@anl.gov",
    description="differentiable models of the SMHM",
    packages=find_packages(),
    url="https://github.com/ArgonneCPAC/diffsmhm"
)
