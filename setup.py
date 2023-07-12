import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='classificationtoolkit',
    version='0.0.3',
    author='Michael Scriney',
    author_email='michael.scriney@dcu.ie',
    description='Toolkit for running multiple classifiers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/scrineym/classificationtoolkit',
    project_urls = {
        "Bug Tracker": "https://github.com/scrineym/classificationtoolkit/issues"
    },
    license='MIT',
    packages=['classificationtoolkit'],
    install_requires=['numpy',
                      'pandas',
                      'seaborn',
                      'scipy',
                      'scikit-learn',
                      'hyperopt',
                      'xgboost',
                      'imbalanced-learn',
                      'tqdm'
                      ],
)