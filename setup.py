from setuptools import setup, find_packages

setup(
  name = 'smart-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.2',
  license='MIT',
  description = 'SMART Fine-Tuning - Pytorch',
  author = 'Flavio Schneider',
  author_email = 'archinetai@protonmail.com',
  url = 'https://github.com/archinetai/smart-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'fine-tuning',
    'pre-trained',
  ],
  install_requires=[
    'torch>=1.6',
    'data-science-types>=0.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)