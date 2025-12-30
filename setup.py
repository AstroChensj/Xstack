from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
import os
import shutil

with open("VERSION") as f:
    lines = f.readlines()
    version = lines[0].strip()
    lastupdate = lines[1].strip()

class Copy_VERSION(_build_py):
    def run(self):
        for pkg in ["Xstack","Xstack_scripts"]:
            dst = os.path.join(pkg,"VERSION")
            shutil.copyfile("VERSION",dst)
        super().run()

setup(
    name="Xstack",
    version=version,
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
        "Xstack": ["data/*.txt","simu/fkspec_sh/*.sh","VERSION"],
        "Xstack_scripts": ["VERSION"]
    },
    entry_points={
        "console_scripts": [
            "runXstack=Xstack_scripts.Xstack_autoscript:main",
        ]
    },
    cmdclass={'build_py': Copy_VERSION},
)
