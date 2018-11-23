import os, re, sys

try:
  from sphinx import version_info as _sphinx_version_info
except ImportError:
  _sphinx_version_info = ()

sys.path.insert(0, '..')

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.napoleon',
  'sphinx.ext.intersphinx',
]

master_doc = 'index'

project = 'povplot'
copyright = '2018, Evalf'

with open(os.path.join('..', 'povplot.py')) as f:
  release = next(filter(None, map(re.compile("^version = '([a-zA-Z0-9.]+)'$").match, f))).group(1)
version = re.search('^[0-9]+\\.[0-9]+', release).group(0)

autodoc_member_order = 'bysource'
if _sphinx_version_info >= (1,8):
    autodoc_default_options = {'members':None, 'show-inheritance':None}
else:
    autodoc_default_flags = ['members', 'show-inheritance']
autodoc_inherit_docstrings = False # i.e. don't document implementations of abstract methods (if the implementation does not have a docstring)

intersphinx_mapping = {
  'python': ('https://docs.python.org/3', None),
  'numpy': ('https://docs.scipy.org/doc/numpy/', None),
  'matplotlib': ('https://matplotlib.org/', None),
}

# vim: sts=2:sw=2:et
