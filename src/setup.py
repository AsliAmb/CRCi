from setuptools import setup, find_packages

setup(
    name='CRCi',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Python package for CMS classification using 29 PSI values',
    install_requires=[
        'joblib', 'pandas','numpy'
    ],
    package_data={
        'CRCi': ['models/*.joblib'],
    },
)


