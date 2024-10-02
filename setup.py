from setuptools import setup

setup(
    name='relchanet',
    version='0.1.0',
    author='Felix Zimmer',
    author_email='felix.zimmer@mail.de',
    description='Neural Network Feature Selection using Relative Change Scores',
    url='https://github.com/flxzimmer/relchanet',
    packages=['relchanet'],
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
