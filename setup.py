from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

setup(
    name="Xstack",
    version="1.1.2",
    description="An X-ray Spectral Shifting and Stacking Code",
    author="Shi-Jiang Chen, Johannes Buchner and Teng Liu",
    author_email="JohnnyCsj666@gmail.com",
    url="https://github.com/AstroChensj/Xstack.git",
    #packages=find_packages(),
    packages=["Xstack","Xstack/utils","Xstack/visual","Xstack/simu","Xstack_scripts"],
    install_requires=[
        "astropy",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "tqdm",
        "numba",
        "sfdmap",
        "joblib",
        "psutil",
    ],
    package_data={
        "Xstack": ["data/*.txt","simu/fkspec_sh/*.sh"],
    },
    entry_points={
        "console_scripts": [
            "runXstack=Xstack_scripts.Xstack_autoscript:main",
            "clear_rf_files=Xstack_scripts.clear_rf_files:main",
        ]
    },
)
