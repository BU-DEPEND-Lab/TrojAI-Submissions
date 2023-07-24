# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['depend']

package_data = \
{'': ['*']}

install_requires = \
['captum>=0.6.0,<0.7.0',
 'mlflow>=2.5.0,<3.0.0',
 'optuna>=3.2.0,<4.0.0',
 'torch==2.0.1',
 'wandb>=0.15.5,<0.16.0']

setup_kwargs = {
    'name': 'depend',
    'version': '0.1.0',
    'description': "'trojai project lib'",
    'long_description': '# DEPEND package for TrojAI Project',
    'author': 'zwc662',
    'author_email': 'zwc662@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)

