import unittest, tempfile, io, collections, pathlib
import numpy, matplotlib.image, matplotlib.cm, matplotlib.figure, matplotlib.backends.backend_agg
import povplot

class common:

  def render_square(self, *, focal_length=50, **kwargs):
    defaults = dict(vertices=[[-9,-9,focal_length],[-9,9,focal_length],[9,-9,focal_length],[9,9,focal_length]],
                    triangles=[[0,1,2],[1,3,2]],
                    values=[0,0,2,2],
                    size=(36,24),
                    camera=dict(location=(0,0,0),look_at=(0,0,focal_length),focal_point=(0,0,focal_length),focal_length=focal_length))
    return self.render(**collections.ChainMap(kwargs, defaults))

  def render_triangle(self, *, focal_length=50, **kwargs):
    defaults = dict(vertices=[[-9,-9,focal_length],[-9,9,focal_length],[9,0,focal_length]],
                    triangles=[[0,1,2]], values=[0,1,2],
                    size=(36,24),
                    camera=dict(location=(0,0,0),look_at=(0,0,focal_length),focal_point=(0,0,focal_length),focal_length=focal_length))
    return self.render(**collections.ChainMap(kwargs, defaults))

  def test_focal_length(self):
    for focal_length in 25, 50:
      with self.subTest(focal_length=focal_length):
        im = self.render_square(focal_length=focal_length)[:,:,:3]
        mask = numpy.zeros([24,36], dtype=bool)
        mask[3:21,9:27] = True
        self.assertTrue(numpy.equal(im[~mask], [[0,0,0]]).all())
        self.assertFalse(numpy.equal(im[mask], [[0,0,0]]).all())

  def test_transparent(self):
    for transparent in True, False:
      with self.subTest(transparent=transparent):
        im = self.render_square(transparent=transparent)
        self.assertEqual(im[0,0,3], 0 if transparent else 255)
        self.assertEqual(im[12,18,3], 255)

  def test_antialias(self):
    for antialias in True, False:
      with self.subTest(antialias=antialias):
        im = self.render_triangle(antialias=antialias, transparent=True)
        if antialias:
          self.assertGreater(len(numpy.unique(im[:,:,3].ravel())), 2)
        else:
          self.assertEqual(numpy.unique(im[:,:,3].ravel()).tolist(), [0, 255])

  def test_cmap_jet(self):
    im = self.render_square(cmap='jet')
    cmap = matplotlib.cm.get_cmap('jet')
    self.assertLess(abs((im[12,9:-9])-cmap((numpy.arange(18)+0.5)/18)*255).max(), 5)

  def test_vmin_vmax(self):
    im = self.render_square(vmin=-1, vmax=3)
    cmap = matplotlib.cm.get_cmap(None)
    self.assertLess(abs((im[12,9:-9])-cmap((numpy.arange(9,27)+0.5)/36)*255).max(), 5)

  def test_auto_camera(self):
    im = self.render_square(camera=None)
    mask = numpy.ones(im.shape[:2], dtype=bool)
    mask[1:-1,1:-1] = False
    self.assertGreater(numpy.max(im[:,:,:3]), 0, msg='object out of sight')
    self.assertTrue(numpy.equal(im[mask][:,:3], [[0,0,0]]).all(), msg='object partly out of sight')

  def test_nprocs(self):
    im = self.render_square(nprocs=2)

class render_triplot(unittest.TestCase, common):

  test_args = dict(vertices=[[-9,-9,50],[-9,9,50],[9,-9,50],[9,9,50]],
                   triangles=[[0,1,2],[1,3,2]], values=[0,1,2,3],
                   size=(36,24))

  png_header = b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a'
  jpg_header = b'\xff\xd8\xff'

  def render(self, **kwargs):
    with tempfile.TemporaryFile('w+b') as f:
      povplot.render_triplot(f, imgtype='png', **kwargs)
      f.flush()
      f.seek(0)
      im = (matplotlib.image.imread(f, format='png')*255).round().astype(numpy.uint8)
      if im.shape[2] == 3:
        im = numpy.concatenate([im, numpy.full_like(im[:,:,:1], 255)], axis=2)
      return im

  def test_guess_imgtype_known(self):
    for header, *suffixes in (self.png_header, '.png', b'.png'),:# (self.jpg_header, '.jpg'):
      for suffix in suffixes:
        with self.subTest(suffix=suffix):
          with tempfile.NamedTemporaryFile('w+b', suffix=suffix) as f:
            povplot.render_triplot(f, **self.test_args)
            f.flush()
            f.seek(0)
            self.assertEqual(f.read(len(header)), header)

  def test_guess_imgtype_unknown(self):
    with tempfile.NamedTemporaryFile('wb', suffix='.unknown') as f:
      with self.assertRaises(ValueError):
        povplot.render_triplot(f, **self.test_args)

  def test_guess_imgtype_no_name(self):
    with io.BytesIO() as f:
      with self.assertRaises(ValueError):
        povplot.render_triplot(f, **self.test_args)

  def test_write_stringio(self):
    with io.BytesIO() as f:
      povplot.render_triplot(f, imgtype='png', **self.test_args)
      f.flush()
      f.seek(0)
      self.assertEqual(f.read(len(self.png_header)), self.png_header)

  def test_write_str(self):
    with tempfile.NamedTemporaryFile('w+b', suffix='.png') as f:
      povplot.render_triplot(f.name, **self.test_args)
      f.flush()
      f.seek(0)
      self.assertEqual(f.read(len(self.png_header)), self.png_header)

class triplot(unittest.TestCase, common):

  def render(self, *, size, **kwargs):
    dpi = 100
    figsize = size[0]/dpi, size[1]/dpi
    with tempfile.TemporaryFile('w+b') as f:
      fig = matplotlib.figure.Figure(figsize=figsize, dpi=dpi)
      matplotlib.backends.backend_agg.FigureCanvas(fig) # sets reference via fig.set_canvas
      ax = fig.add_axes([0,0,1,1])
      ax.axis('off')
      povplot.triplot(ax, **kwargs)
      savefig_kwargs = dict(facecolor='none', edgecolor='none') if kwargs.get('transparent', False) else {}
      fig.savefig(f, format='png', **savefig_kwargs)
      fig.set_canvas(None) # break circular reference
      f.flush()
      f.seek(0)
      return (matplotlib.image.imread(f, format='png')*255).round().astype(numpy.uint8)

class render(unittest.TestCase):

  def test_error(self):
    with tempfile.TemporaryFile('wb') as f:
      with self.assertRaisesRegex(povplot.PovrayError, '^Povray failed with code -?[0-9]+$') as cm:
        povplot.render(f, scene='invalid', size=(36,24), imgtype='png')
      with self.subTest('rendered script'):
        self.assertEqual(cm.exception.rendered_script, 'invalid')

class overlay_colorbar(unittest.TestCase):

  def test(self):
    # Test if code runs only.
    dpi = 100
    with tempfile.TemporaryFile('w+b') as f:
      fig = matplotlib.figure.Figure()
      matplotlib.backends.backend_agg.FigureCanvas(fig) # sets reference via fig.set_canvas
      ax = fig.add_axes([0,0,1,1])
      im = povplot.triplot(ax,
                           hide_frame=True,
                           vertices=[[-9,-9,50],[-9,9,50],[9,-9,50],[9,9,50]],
                           triangles=[[0,1,2],[1,3,2]], values=[0,1,2,3])
      povplot.overlay_colorbar(fig, im)
      fig.savefig(f, format='png', facecolor='none', edgecolor='none')
      fig.set_canvas(None) # break circular reference
      f.flush()
      f.seek(0)

class sphinx(unittest.TestCase):

  def test(self):
    with tempfile.TemporaryDirectory(prefix='povplot-') as tmpdir:
      tmpdir = pathlib.Path(tmpdir)
      root = pathlib.Path(__file__).parent
      try:
        from sphinx.application import Sphinx
      except ImportError:
        self.skipTest()
        raise ValueError
      app = Sphinx(srcdir=str(root/'docs'),
                   confdir=str(root/'docs'),
                   outdir=str(tmpdir/'html'),
                   doctreedir=str(tmpdir/'doctree'),
                   buildername='html',
                   freshenv=True,
                   warningiserror=True,
                   confoverrides=dict(nitpicky=True))
      app.build()
      if app.statuscode:
        self.fail('sphinx build failed with code {}'.format(app.statuscode))

# vim: sts=2:sw=2:et
