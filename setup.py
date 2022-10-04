import setuptools

setuptools.setup(
    name="relativebilbying",
    version="0.0.1",
    author="Justin Janquart, Anna Puecher",
    author_email="j.janquart@uu.nl",
    description="Code to use relative binning in bilby",
    url="TBD",
    packages=["relativebilbying"],
    install_requires=["bilby",
                      "numpy",
                      "lalsuite"],
    classifiers=["Development Status :: 3 - Alpha",
                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Programming Language :: Python :: 3.9",
                 "License :: OSI Approved :: MIT License"],
    python_requires = ">=3.7"
)