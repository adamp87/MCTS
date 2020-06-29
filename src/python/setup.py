from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='Alpha4',
   version='0.1',
   description='',
   author='AdamP',
   packages=['Alpha4'],
   install_requires=required,  # external packages as dependencies
)
