"""Python setup.py for padlock_env package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("padlock_env", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="padlock_env",
    version=read("padlock_env", "VERSION"),
    description="Awesome padlock_env created by cyprienc",
    url="https://github.com/cyprienc/padlock-env/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="cyprienc",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["padlock_env = padlock_env.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
