from setuptools import find_packages, setup
import pathlib

parent_path = str(pathlib.Path(__file__).parent)
absolute_path = parent_path + '/requirements.txt'
with open(absolute_path) as f:
    required = f.read().splitlines()

setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='ML in prod: Homework 1',
    author='Kadiev Alan',
    install_requires=required,
    license='TechnoPark',
)