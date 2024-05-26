from setuptools import setup, find_packages

setup(
    name="nimbus",
    author="Anna M. Diaz-Rovira",
    author_email="annadiarov@gmail.com",
    description=("NIMBUS: Neoantigen Immunogenicity Multimodal Prediction "
                 "Utilizing Surface-informed Fingerprints."),
    # license="MIT",
    keywords="cancer vaccines, neoantigens, immunogenicity, machine learning",
    url="https://github.com/annadiarov/nimbus",
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)