from setuptools import find_packages, setup
from typing import List

def get_requirements( file_path:str)-> List[str]
    """
    This function returns the list of requirements
    """

    requirements= []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [ package.replace("\n","")for package in requirements]

    if "-e ." in requirements:
        requirements.remove("-e .")
    return requirements 

setup(
    name="employee_attrition_project",
    version="0.0.1",
    author="Samuel Cuevas",
    author_email="samidelcueva@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)