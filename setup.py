from setuptools import setup, find_packages

setup(
      name='cosmax',
      version='0.1',
      author='Andrin Rehmann',
      author_email='andrinrehmann@gmail.com',
      description='Fast and differentiable implementations of operations needed for inference and analysis in cosmology. Powered by JAX.',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/andrinr/cosmax',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.6',
      install_requires=[
          # List your package dependencies here
      ],
  )