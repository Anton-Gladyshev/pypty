from setuptools import setup
setup(name="pypty",
    version='2.2',
    description="A CuPy-based set of scripts for multisclie reconstrcutions on a GPU. Can be configured for STEM (ptychograpgy), TEM (tilt or focal series) or pretty much any combination of this things.",
    packages=['pypty'],
    author="Anton Gladyshev",
    install_requires=[
        "numpy",
        "h5py",
        "matplotlib",
        "scipy",
        "scikit-image",
        "tqdm",
    ],
    extras_require={
        "gpu": ["cupy"],
    },
    zip_safe=False)

