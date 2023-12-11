from setuptools import setup

with open("README.md", "r") as file_:
    project_description = file_.read()

with open("requirements.txt", "r") as file_:
    project_requirements = file_.read().split("\n")

setup(
    name="ftracking",
    version="0.1.0",
    description="Tracking fishes in video, with and without stereo setup",
    license="MIT",
    long_description=project_description,
    author="Fatemeh Karimi Nejadasl",
    author_email="fkariminejadasl@gmail.com",
    url="https://github.com/fkariminejadasl/tracking",
    packages=["tracking"],
    install_requires=project_requirements,
)
