from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
    return requirements

setup(
    name='internproject',
    version='0.0.1',
    packages=find_packages(),   # <-- include your packages
    install_requires=get_requirements('requirements.txt'),
)