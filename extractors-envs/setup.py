from setuptools import setup, find_packages

setup(name='extractors',
      version='0.0.1',
      description='The Extractors Environment',
      author='H Zheng',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl']
)
