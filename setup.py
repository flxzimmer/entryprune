from setuptools import setup

setup(
    name='entryprune',
    version='0.1.0',
    author='Felix Zimmer',
    author_email='felix.zimmer@mail.de',
    description='EntryPrune: Neural Network Feature Selection using First Impressions',
    url='https://github.com/flxzimmer/entryprune',
    packages=['entryprune'],
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'seaborn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
