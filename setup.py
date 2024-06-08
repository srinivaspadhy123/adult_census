from setuptools import setup,find_packages

setup(
    name="Adult_Census_Salary_Prediction",
    version='0.0.1',
    author="Srinivas Padhy",
    author_email="srinivaspadhybm@gmial.com",
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)