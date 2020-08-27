import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MABtest", # Replace with your own username
    version="1.0",
    author="Jinjin Tian",
    author_email="jinjint@andrew.com",
    description="An open source package for online experiments using Multi-armed bandits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['MABtest'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)