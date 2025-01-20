SYNC_NONE = 0  # no sync: runs very fast, unplayable, music is choppy
SYNC_AUDIO = 1  # sync to audio: rate is perfect, can glitch sometimes, screen tearing can be bad
SYNC_PYGAME = 2  # sync to pygame's clock, adaptive audio: generally reliable, some screen tearing
SYNC_VSYNC = 3  # sync to external vsync, adaptive audio: requires ~60Hz vsync, no tearing

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

extensions = [Extension("cycore.*", ["nes/cycore/*.pyx"], include_dirs=[numpy.get_include()])]
extensions = cythonize(extensions, compiler_directives={"language_level": 3, "profile": False, "boundscheck": False, "nonecheck": False, "cdivision": True}, annotate=True)


import numpy
print(numpy.get_include())

import pyximport
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

from nes.cycore.system import NES   # make the key NES object available at the top level


