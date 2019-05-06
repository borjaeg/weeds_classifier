import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()

setup(
	name="plant-seedlings-classifier",
	version="0.1",
	description="Weeds Classifier",
	long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/borjaeg/weedsapp",
    author="Borjakas",
    author_email="borja.espejo.garcia@gmail.com",
    license="MIT",
    packages=["weeds_classifier"],
    include_package_data=True,
    install_requires=list_reqs()
)

#["keras", "opencv-python", "scikit-learn", "pandas", "numpy"]