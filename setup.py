from setuptools import setup
import os, re

with open('povplot.py') as f:
  version = next(filter(None, map(re.compile("^version = '([a-zA-Z0-9.]+)'$").match, f))).group(1)

setup(
  name = 'povplot',
  version = version,
  description = 'A library for rendering triangular grids with Povray',
  author = 'Evalf',
  author_email = 'info@evalf.com',
  url = 'https://github.com/evalf/povplot',
  download_url = 'https://github.com/evalf/povplot/releases',
  py_modules = ['povplot'],
  license = 'MIT',
  python_requires = '>=3.5',
  install_requires = ['jinja2','numpy','matplotlib'],
  extras_require = dict(
    docs=['Sphinx'],
  )
)

# vim: sts=2:sw=2:et
