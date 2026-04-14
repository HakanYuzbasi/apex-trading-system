from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Ensure the working directory is the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

extensions = [
    Extension(
        "quant_system.execution.fast_math",
        ["quant_system/execution/fast_math.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="FastMath",
    ext_modules=cythonize(extensions, language_level=3),
)
