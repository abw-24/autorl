from setuptools import setup

setup(
   name='deepRL',
   version='0.1',
   description='Utitlies for building and training neural RL agents',
   author='Andrew West',
   packages=['deepRL'],  #same as name
   install_requires=['tensorflow'], #external packages as dependencies
)