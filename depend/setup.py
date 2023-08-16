# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['depend',
 'depend.core',
 'depend.core.dependent',
 'depend.core.learners',
 'depend.core.logger',
 'depend.lib',
 'depend.lib.utils',
 'depend.pipelines.input_attribution',
 'depend.utils']

package_data = \
{'': ['*']}

install_requires = \
['captum>=0.6.0,<0.7.0',
 'datasets>=2.14.0,<3.0.0',
 'pyarrow>=12.0.1,<13.0.0',
 'pydantic>=2.0.3,<3.0.0',
 'torch-ac==1.1.0',
 'torch==2.0.1',
 'torchvision>=0.15.2,<0.16.0']

setup_kwargs = {
    'name': 'depend',
    'version': '0.1.0',
    'description': "'trojai project lib'",
    'long_description': '# DEPEND package for TrojAI Project\n\n* Install the project by `python -`\n* Use `poetry cache clear --all .` to clean poetry cache if poetry takes too long to search for a library',
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

