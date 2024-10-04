from setuptools import setup, find_packages
HYPEN_E_DOT = "-e ."
def get_requirements()->list:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
        requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='ml_prj_structure',
    version='0.1',
    packages=find_packages(),
    author='Dhruv Vaisnav'  ,
    author_email='dhruvvaishnav2125@gmail.com',
    description='A project structure for machine learning projects',
    install_requires=get_requirements(),
)