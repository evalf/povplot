# Copyright (c) 2018 Evalf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
povplot
=======

A library for rendering triangular grids with Povray.

The functions :func:`tripcolor` (for matplotlib figures) and
:func:`render_tripcolor` (standalone) render a 3D triangular grid using Povray,
with an interface similar to :meth:`matplotlib.axes.Axes.tripcolor`.  If you
want more control over the scene, use :func:`render` (standalone).
'''

version = '1.0b0'

import tempfile, subprocess, contextlib, os, io
import jinja2, numpy, matplotlib.image, matplotlib.colors, matplotlib.cm, matplotlib.artist, matplotlib.patches

def _filter_as_vector(data):
  # Jinja filter to convert a 1d numpy array to a povray vector.
  data = numpy.asarray(data)
  assert data.ndim == 1
  return '<{}>'.format(','.join(map(str, data)))

def _filter_as_vector_list(data, pad_length=0):
  # Jinja filter to convert a 2d numpy array (or 1d if pad_length is nonzero)
  # to a povray vector list.  If the second dimension is smaller than
  # pad_length or absent, the array is padded with zeros.
  assert not isinstance(data, jinja2.Undefined)
  data = numpy.asarray(data)
  if pad_length and data.ndim == 1:
    fmt_vec = ['{0}']+['0']*(pad_length-1)
  else:
    assert data.ndim == 2
    fmt_vec = ['{{0[{}]}}'.format(i) for i in range(data.shape[1])]
  fmt_vec.extend(['0']*max(0, (pad_length-len(fmt_vec))))
  fmt_vec = ',<{}>'.format(','.join(fmt_vec))
  return '{{{}{}}}'.format(len(data), ''.join(map(fmt_vec.format, data)))

def _filter_cmap_to_pigment(cmap, direction='x'):
  # Jinja filter to convert a matplotlib cmap to a povray pigment.
  cmap = matplotlib.cm.get_cmap(cmap)
  fmt = '[{0:.4f} color srgb<{1[0]:.3f},{1[1]:.3f},{1[2]:.3f}>]'.format
  if isinstance(cmap, matplotlib.colors.ListedColormap):
    str_cmap = ' '.join(map(fmt, numpy.linspace(0, 1, len(cmap.colors)), cmap.colors))
  elif isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
    colors = cmap(numpy.arange(256))
    str_cmap = ' '.join(map(fmt, numpy.linspace(0, 1, len(colors)), colors))
  else:
    raise ValueError
  return 'pigment {{ gradient {} color_map {{ {} }} }}'.format(direction, str_cmap)

@jinja2.environmentfilter
def _filter_equivalent_focal_length_to_angle(env, focal_length):
  w, h = env.globals['size']
  # Scaling of the resulution such that the diagonal matches a standard 35mm
  # film, 36x24mm.  See
  # https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
  scale = ((36**2+24**2)/(w**2+h**2))**0.5 # unit: mm/px
  # Compute the horizontal angle in degrees based on an equivalent film width.
  equiv_w = w*scale
  return 2*numpy.arctan(equiv_w/(2*focal_length))*180/numpy.pi

def _get_norm(norm, vmin=None, vmax=None):
  if not norm:
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  return norm

def _test_None_or_undefined(value):
  return value is None or isinstance(value, jinja2.Undefined)

_module_povplot = '''\
{% macro global_settings(assumed_gamma=1.0, mm_per_unit=10) %}
  global_settings {
    assumed_gamma {{ assumed_gamma }}
    mm_per_unit {{ mm_per_unit }}
    {% if caller is defined %}
      {{ caller() }}
    {% endif %}
  }
{% endmacro %}
{% macro camera(location=(0,0,0), sky=(0,1,0), look_at=(0,0,1), focal_point=(0,0,1), focal_length=50) %}
  camera {
    location {{ location | as_vector }}
    sky {{ sky | as_vector }}
    look_at {{ look_at | as_vector }}
    focal_point {{ focal_point | as_vector }}
    angle {{ focal_length | equivalent_focal_length_to_angle }}
    up y
    right {{ aspect_ratio }}*x
    {% if caller is defined %}
      {{ caller() }}
    {% endif %}
  }
{% endmacro %}
{% macro light_source_point(position=(0,1,0), color='srgb 1') %}
  light_source {
    {{ position | as_vector }}
    {{ color }}
    {% if caller is defined %}
      {{ caller() }}
    {% endif %}
  }
{% endmacro %}
{% macro mesh2(vertices, triangles, normals=None, uv=None) %}
  mesh2 {
    vertex_vectors {{ vertices | as_vector_list }}
    {% if normals is not None_or_undefined %}
      normal_vectors {{ normals | as_vector_list }}
    {% endif %}
    {% if uv is not None_or_undefined %}
      uv_vectors {{ uv | as_vector_list(pad_length=2) }}
    {% endif %}
    face_indices {{ triangles | as_vector_list }}
    {% if uv is not None_or_undefined %}
      uv_mapping
    {% endif %}
    {% if caller is defined %}
      {{ caller() }}
    {% endif %}
  }
{% endmacro %}
{% macro tripcolor(vertices, triangles, normals=None, values=None, cmap=None, norm=None, vmin=None, vmax=None) %}
  {% call mesh2(vertices, triangles, normals, uv=get_norm(norm, vmin, vmax)(values)) %}
    texture {
      {{ cmap | cmap_to_pigment }}
      finish { ambient 0.5 diffuse 0.5 emission 0 }
    }
    {% if caller is defined %}
      {{ caller() }}
    {% endif %}
  {% endcall %}
{% endmacro %}
'''

def get_env():
  env = jinja2.Environment()
  env.filters.update(
    as_vector=_filter_as_vector,
    as_vector_list=_filter_as_vector_list,
    cmap_to_pigment=_filter_cmap_to_pigment,
    equivalent_focal_length_to_angle=_filter_equivalent_focal_length_to_angle)
  env.tests.update(None_or_undefined=_test_None_or_undefined)
  env.globals.update(get_norm=_get_norm)
  env.globals.update(povplot=env.from_string(_module_povplot))
  return env

_os_fspath = getattr(os, 'fspath', lambda x: x)

def _guess_imgtype(file):
  try:
    file = _os_fspath(file)
  except TypeError:
    pass
  if isinstance(file, str):
    name = file
  elif isinstance(file, bytes):
    name = file.decode(errors='ignore')
  elif hasattr(file, 'name'):
    return _guess_imgtype(file.name)
  else:
    raise ValueError('cannot find the filename of {!r}'.format(file))
  imgtype = os.path.splitext(name)[1]
  if imgtype not in {'.png'}:
    raise ValueError('unknown image type: {}'.format(imgtype))
  return imgtype[1:]

@contextlib.contextmanager
def _ensure_writeable_fd(file):
  if hasattr(file, 'fileno'):
    try:
      fileno = file.fileno()
    except io.UnsupportedOperation:
      fileno = None
    if isinstance(fileno, int):
      yield fileno
    else:
      with tempfile.TemporaryFile('r+b') as intermediate:
        yield intermediate.fileno()
        intermediate.seek(0)
        while True:
          chunk = intermediate.read(4096)
          if not chunk:
            break
          while chunk:
            n = file.write(chunk)
            chunk = chunk[n:]
  else:
    with open(file, 'wb') as f:
      yield f.fileno()

def render(dst, *, scene, size, antialias=False, transparent=False, scene_args=None, nprocs=None, imgtype=None):
  '''Render ``scene`` to ``dst`` using Jinja2 and Povray.

  ``scene`` should be a valid Povray script.  The script is preprocessed with
  Jinja2 with dictionary ``scene_args`` as context.  Jinja2 is preloaded with a
  ``povplot`` module containing the following macros:

  .. function:: povplot.global_settings(assumed_gamma=1.0, mm_per_unit=10)

      A `global_settings <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_1>`_
      statement with the following parameters.

      :type assumed_gamma: :class:`float`, default: ``1.0``
      :param assumed_gamma: See `assumed_gamma <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_1_3>`_.
      :type mm_per_unit: :class:`float`, default: ``10``
      :param mm_per_unit: See `mm_per_unit <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_1_9>`_.

      Additional statements can be added to the body of the macro.

  .. function:: povplot.camera(location=(0,0,0), sky=(0,1,0), look_at=(0,0,1), focal_point=(0,0,1), focal_length=50)

      A `camera <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2>`_
      statement with the following parameters:

      :type location: 3d vector, default: ``(0,0,0)``
      :param location: The location of the camera.  See `location <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2_1_1>`_.

      :type look_at: 3d vector, default: ``(0,0,1)``
      :param look_at: The point the camera looks at.  See `look_at <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2_1_1>`_.
      :type sky: 3d vector, default: ``(0,1,0)``
      :param sky: This orientates the camera such that a line from ``location`` to ``sky`` is displayed upward.  See `sky <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2_1_2>`_.
      :type focal_point: 3d vector, default: ``(0,0,1)``
      :param focal_point: The focal point of the camera.  See `focal_point <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2_3>`_.
      :type focal_length: :class:`float`, default: ``50``
      :param focal_length: The focal length of the camera.  This defines the `angle <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_2_1_3>`_ parameter based on the `35 mm equivalent focal length <https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length>`_ rule.

  .. function:: povplot.light_source_point(position=(0,1,0), color='srgb 1')

      A simple `point light <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_4_1_1>`_ statement.

      :type position: 3d vector, default: ``(0,1,0)``
      :param position: The position of the point light source.
      :type color: :class:`str`, default: ``'srgb 1'``
      :param color: The color of the point light source.  See `color expressions <http://www.povray.org/documentation/3.7.0/r3_3.html#r3_3_1_7>`_.

  .. function:: povplot.mesh2(vertices, triangles, normals=None, uv=None)

      A triangular grid statement.  Textures can be defined in the body of the macro.

      :type vertices: :class:`numpy.ndarray` with shape ``(nverts,3)``
      :param vertices: The vertices of the triangulation.
      :type triangles: :class:`numpy.ndarray` with shape ``(ntriangles,3)`` and dtype :class:`int`
      :param triangles: The indices of vertices defining the triangles.
      :type normals: :class:`numpy.ndarray` with shape ``(nverts,3)``, optional
      :param normals: The outward normals at the vertices.
      :type uv: :class:`numpy.ndarray` with shape ``(nverts,2)``, optional
      :param uv: UV-coordinate per vertex.  This can be used for texture mapping.  See `uv mapping <http://www.povray.org/documentation/3.7.0/r3_4.html#r3_4_6_7>`_.

  Parameters
  ----------
  dst: :class:`str`, :class:`os.PathLike` or :class:`io.IOBase`
      The destination of the rendered image.  Should be a :class:`str` or
      path-like object, or a file opened for writing binary data.
  scene: :class:`str`
      The Povray scene to render.  The scene is preprocessed using Jinja2
      with ``scene_args``.
  size: :class:`tuple` of two :class:`int`\\s
      The size (width, height) of the rendered image.
  antialias: :class:`bool`, default: ``False``
      Let Povray produce an antialiased image.
  transparent: :class:`bool`, default: ``False``
      Let Povray produce an image with transparency.
  scene_args: :class:`collections.abc.Mapping`, optional
      Dictionary of scene arguments passed to Jinja2 when rendering the
      scene.
  nprocs: :class:`int`, default: ``1``
      Number of threads Povray may use to render the image.
  imgtype: :class:`str`, default: ``'png'``
      The type of the rendered image: only ``'png'`` is suppored.  If
      absent the type is determined based on ``dst``.

  Raises
  ------
  :class:`PovrayError`
      If Povray returns an error.
  '''

  if imgtype is None:
    imgtype = _guess_imgtype(dst)

  with tempfile.NamedTemporaryFile('w', suffix='.pov') as f_src:
    env = get_env()
    env.globals.update(size=size, aspect_ratio=size[0]/size[1])
    template = env.from_string(scene)
    template.stream(scene_args or {}).dump(f_src)
    f_src.flush()

    flag = lambda value, name: ('+' if value else '-')+name
    povray_args = ['povray', '-D', flag(antialias,'A'), flag(transparent,'UA'), '+W{}'.format(size[0]), '+H{}'.format(size[1]), '+I'+f_src.name, '+F{}'.format({'png': 'N'}[imgtype]), '+O-']
    if nprocs:
      povray_args + ['+wt{}'.format(nprocs)]
    with _ensure_writeable_fd(dst) as fd_dst:
      p_povray = subprocess.run(povray_args, stdin=subprocess.DEVNULL, stdout=fd_dst, stderr=subprocess.PIPE)
    if p_povray.returncode:
      raise PovrayError(p_povray.returncode, p_povray.stderr.decode(errors='ignore'), template, scene_args)

def render_tripcolor(dst, *, size, vertices, triangles, values, cmap=None, norm=None, vmin=None, vmax=None, normals=None, camera=None, mm_per_unit=10, antialias=False, transparent=False, nprocs=None, imgtype=None):
  '''Render a triangular grid with Povray.

  Parameters
  ----------
  dst: :class:`str`, :class:`os.PathLike` or :class:`io.IOBase`
      The destination of the rendered image.  Should be a :class:`str` or
      path-like object, or a file opened for writing binary data.
  size: :class:`tuple` of two :class:`int`\\s
      The size (width, height) of the rendered image.
  vertices: :class:`numpy.ndarray` with shape ``(nverts,3)``
      The vertices of the triangulation.
  triangles: :class:`numpy.ndarray` with shape ``(ntriangles,3)`` and dtype :class:`int`
      The indices of vertices defining the triangles.
  values: :class:`numpy.ndarray` with shape ``(nverts,)``
      The values of the vertices.  These will be mapped to a color.
  normals: :class:`numpy.ndarray` with shape ``(nverts,3)``, optional
      The outward normals at the vertices.
  cmap: :class:`str` or :class:`matplotlib.colors.Colormap`, optional
      The name or an instance of a matplotlib colormap to be applied to the
      normalized ``values``.
  norm: :class:`matplotlib.colors.Normalize`, optional
      The normalization of ``values`` to the range [0,1], applied before
      colormapping.
  vmin: :class:`float`, optional
      The minimum value of the normalization.
  vmax: :class:`float`, optional
      The maximum value of the normalization.
  camera: :class:`collections.abc.Mapping`, optional
      The position and orientation of the camera.  If absent the camera is
      positioned such that the triangulation is completely in view.  While
      optional, users are encouraged to define the camera manually as the
      autopositioning may change in the future.  See the Jinja2 macro
      :func:`povplot.camera` for the possible settings.
  mm_per_unit: :class:`float`, default: ``10``
  antialias: :class:`bool`, default: ``False``
      Let Povray produce an antialiased image.
  transparent: :class:`bool`, default: ``False``
      Let Povray produce an image with transparency.
  nprocs: :class:`int`, default: ``1``
      Number of threads Povray may use to render the image.
  imgtype: :class:`str`, default: ``'png'``
      The type of the rendered image: only ``'png'`` is suppored.  If
      absent the type is determined based on ``dst``.

  Raises
  ------
  :class:`PovrayError`
      If Povray returns an error.

  Example
  -------

  The following example renders a unit square, subdivided in two triangles, with a
  linear color gradient:

  >>> render_tripcolor('example.png',
  ...                  vertices=[[0,0,0],[0,1,0],[1,0,0],[1,1,0]],
  ...                  triangles=[[0,1,2],[1,3,2]],
  ...                  values=[0,1,2,3],
  ...                  size=(800,600))
  '''

  tripcolor = dict(vertices=vertices, triangles=triangles, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, values=values)
  if normals is not None:
    tripcolor['normals'] = normals

  if not camera:
    bbox_min, bbox_max = numpy.min(vertices, axis=0), numpy.max(vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    bbox_size = numpy.max(bbox_max - bbox_min, axis=0)
    location = center + numpy.array([0.5,0.3,0.8])*3*bbox_size
    sky = center + numpy.array([0,1,0])*10*bbox_size
    camera = dict(look_at=center, focal_point=center, sky=sky, location=location)

  global_settings = dict(mm_per_unit=mm_per_unit)
  scene_args = dict(tripcolor=tripcolor, camera=camera, light=camera.get('sky', (0,1,0)), global_settings=global_settings)
  scene = '''\
    #version 3.7;
    {% import povplot as povplot %}
    {{ povplot.global_settings(**global_settings) }}
    {{ povplot.camera(**camera) }}
    {{ povplot.light_source_point(light) }}
    {{ povplot.tripcolor(**tripcolor) }}
  '''

  render(dst, imgtype=imgtype, size=size, antialias=antialias, transparent=transparent, nprocs=nprocs, scene_args=scene_args, scene=scene)

def tripcolor(ax, *, vertices, triangles, values, normals=None, cmap=None, norm=None, vmin=None, vmax=None, camera=None, mm_per_unit=10, antialias=False, transparent=False, nprocs=None, hide_frame=False, hide_ticks=True):
  '''Plot a triangular grid in a matplotlib axes.

  Parameters
  ----------
  ax: :class:`matplotlib.axes.Axes`
      The matplotlib axes to render into.
  vertices: :class:`numpy.ndarray` with shape ``(nverts,3)``
      The vertices of the triangulation.
  triangles: :class:`numpy.ndarray` with shape ``(ntriangles,3)`` and dtype :class:`int`
      The indices of vertices defining the triangles.
  values: :class:`numpy.ndarray` with shape ``(nverts,)``
      The values of the vertices.  These will be mapped to a color.
  normals: :class:`numpy.ndarray` with shape ``(nverts,3)``, optional
      The outward normals at the vertices.
  cmap: :class:`str` or :class:`matplotlib.colors.Colormap`, optional
      The name or an instance of a matplotlib colormap to be applied to the
      normalized ``values``.
  norm: :class:`matplotlib.colors.Normalize`, optional
      The normalization of ``values`` to the range [0,1], applied before
      colormapping.
  vmin: :class:`float`, optional
      The minimum value of the normalization.
  vmax: :class:`float`, optional
      The maximum value of the normalization.
  camera: :class:`collections.abc.Mapping`, optional
      The position and orientation of the camera.  If absent the camera is
      positioned such that the triangulation is completely in view.  While
      optional, users are encouraged to define the camera manually as the
      autopositioning may change in the future.  See the Jinja2 macro
      :func:`povplot.camera` for the possible settings.
  mm_per_unit: :class:`float`, default: ``10``
  antialias: :class:`bool`, default: ``False``
      Let Povray produce an antialiased image.
  transparent: :class:`bool`, default: ``False``
      Let Povray produce an image with transparency.
  nprocs: :class:`int`, default: ``1``
      Number of threads Povray may use to render the image.
  hide_frame: :class:`bool`, default: ``False``
      Hide the frame of the axes ``ax``.
  hide_ticks: :class:`bool`, default: ``True``
      Hide the ticks of the axes ``ax``.

  Returns
  -------
  tripcolor: :class:`AxesTripcolor`
      A matplotlib artist and scalar mappable.

  Raises
  ------
  :class:`PovrayError`
      If Povray returns an error.

  Example
  -------

  The following example renders a unit square, subdivided in two triangles, with a
  linear color gradient:

  >>> import matplotlib.figure
  >>> fig = matplotlib.figure.Figure()
  >>> ax = fig.add_subplot(111)
  >>> im = tripcolor(ax,
  ...                vertices=[[0,0,0],[0,1,0],[1,0,0],[1,1,0]],
  ...                triangles=[[0,1,2],[1,3,2]],
  ...                values=[0,1,2,3])
  >>> fig.colorbar(im, ax=ax)
  '''

  im = AxesTripcolor(
    vertices=vertices, triangles=triangles, values=values, camera=camera,
    antialias=antialias, transparent=transparent, nprocs=nprocs, norm=norm,
    cmap=cmap, mm_per_unit=mm_per_unit)
  if vmin is not None or vmax is not None:
    im.set_clim(vmin, vmax)
  else:
    im.autoscale_None()
  ax.add_image(im)
  if hide_ticks:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  if hide_frame:
    ax.axis('off')
  return im

def overlay_colorbar(fig, sm, *, colorbar_width=0.2, label_width=0.5, margin=0.2, background=(1,1,1,0.5)):
  '''Add a colorbar to a matplotlib figure.

  Parameters
  ----------
  fig: :class:`matplotlib.figure.Figure`
      The figure to add the colorbar to.
  sm: :class:`matplotlib.cm.ScalarMappable`
      The ScalarMappable to which the colorbar applies.
  colorbar_width: :class:`float`, default: ``0.2``
      The width of the colorbar.
  label_width: :class:`float`, default: ``0.5``
      The size reserved for the tick labels.
  margin: :class:`float`, default: ``0.2``
      The margin around the colorbar.
  background: :class:`tuple` of length ``3`` or ``4``, default: ``(1,1,1,0.5)``
      The background of the colorbar axes, margin included.

  Returns
  -------
  cax: :class:`matplotlib.axes.Axes`
      The axes containing the colorbar.
  '''

  width, height = fig.get_size_inches()
  cax = fig.add_axes([1-(margin+label_width+colorbar_width)/width, margin/height, colorbar_width/width, 1-2*margin/height])
  cax.add_patch(matplotlib.patches.Rectangle([1-(2*margin+colorbar_width+label_width)/width, 0], (2*margin+colorbar_width+label_width)/width, 1, transform=fig.transFigure, facecolor=background, zorder=-1, clip_on=False))
  fig.colorbar(sm, cax=cax)
  return cax

class AxesTripcolor(matplotlib.artist.Artist, matplotlib.cm.ScalarMappable):
  '''A matplotlib artist for rendering a triangular grid with Povray.

  Parameters
  ----------
  vertices: :class:`numpy.ndarray` with shape ``(nverts,3)``
      The vertices of the triangulation.
  triangles: :class:`numpy.ndarray` with shape ``(ntriangles,3)`` and dtype :class:`int`
      The indices of vertices defining the triangles.
  values: :class:`numpy.ndarray` with shape ``(nverts,)``
      The values of the vertices.  These will be mapped to a color.
  normals: :class:`numpy.ndarray` with shape ``(nverts,3)``, optional
      The outward normals at the vertices.
  cmap: :class:`str` or :class:`matplotlib.colors.Colormap`, optional
      The name or an instance of a matplotlib colormap to be applied to the
      normalized ``values``.
  norm: :class:`matplotlib.colors.Normalize`, optional
      The normalization of ``values`` to the range [0,1], applied before
      colormapping.
  camera: :class:`collections.abc.Mapping`, optional
      The position and orientation of the camera.  If absent the camera is
      positioned such that the triangulation is completely in view.  While
      optional, users are encouraged to define the camera manually as the
      autopositioning may change in the future.
  mm_per_unit: :class:`float`, default: ``10``
  antialias: :class:`bool`, default: ``False``
      Let Povray produce an antialiased image.
  transparent: :class:`bool`, default: ``False``
      Let Povray produce an image with transparency.
  nprocs: :class:`int`, default: ``1``
      Number of threads Povray may use to render the image.
  '''

  def __init__(self, *, vertices, triangles, values, normals=None, camera=None, antialias=False, transparent=False, nprocs=None, cmap=None, norm=None, mm_per_unit=10):
    matplotlib.artist.Artist.__init__(self)
    matplotlib.cm.ScalarMappable.__init__(self, cmap=cmap, norm=norm)
    self._vertices = vertices
    self._triangles = triangles
    self._normals = normals
    self._camera = dict(camera or {})
    self._mm_per_unit = mm_per_unit
    self._antialias = antialias
    self._transparent = transparent
    self._nprocs = nprocs
    self.set_array(values)

  def draw(self, renderer, *args, **kwargs):
    trans = self.get_transform()
    bbox = trans.transform([(0,0),(1,1)]).round().astype(int)
    bbox.sort(axis=0)
    shape = bbox[1]-bbox[0]

    with tempfile.TemporaryFile('w+b') as f_im:
      render_tripcolor(
        f_im, imgtype='png', size=shape,
        vertices=self._vertices, triangles=self._triangles, normals=self._normals,
        values=self.get_array(), norm=self.norm, cmap=self.cmap,
        camera=self._camera, transparent=self._transparent, antialias=self._antialias,
        nprocs=self._nprocs, mm_per_unit=self._mm_per_unit)
      f_im.flush()
      f_im.seek(0)
      im = (matplotlib.image.imread(f_im, format='png')*255).round().astype(numpy.uint8)
      assert im.shape[:2] == tuple(shape[::-1])
      if im.shape[2] == 3:
        im = numpy.concatenate([im, numpy.full_like(im[:,:,:1], 255)], axis=2)

    gc = renderer.new_gc()
    gc.set_alpha(self.get_alpha())
    gc.set_url(self.get_url())
    gc.set_gid(self.get_gid())
    renderer.draw_image(gc, bbox[0][0], bbox[0][1], im[::-1])
    gc.restore()

class PovrayError(Exception):
  '''Povray error.

  .. attribute:: returncode

      The return code of Povray.

  .. attribute:: stderr

      The captured stderr of Povray.

  .. attribute:: script

      The unrendered script.  See also :attr:`rendered_script`.

  .. attribute:: script_args

      The arguments used to render the script.
  '''

  def __init__(self, returncode, stderr, script, script_args):
    self.returncode = returncode
    self.stderr = stderr
    self.script = script
    self.script_args = script_args
    super().__init__()

  def __str__(self):
    return 'Povray failed with code {}'.format(self.returncode)

  @property
  def rendered_script(self):
    '''The rendered script as passed to Povray.'''

    return self.script.render(self.script_args or {})

# vim: sts=2:sw=2:et
