from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='SPACE2',
    version='0.0.1',
    description='Set of functions to cluster CDR structures',
    license='BSD 3-clause license',
    maintainer=['Brennan Abanades','Fabian Spoendlin'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email=['brennan.abanadeskenyon@stx.ox.ac.uk','fabian.spoendlin@stats.ox.ac.uk'],
    include_package_data=True,
    packages=find_packages(include=('SPACE2', 'SPACE2.*')),
    install_requires=[
        'numba>=0.51.0',
        'joblib>=1.0.0',
        'scipy',
        'pandas',
        'scikit-learn>=1.2.1',
    ],
)
