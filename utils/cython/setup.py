from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the sources
sources = [
    os.path.join(current_dir, 'assemble_matrices.pyx'),
    os.path.join(current_dir, 'K_beam_func.c'),
    os.path.join(current_dir, 'X_beam_func.c'),
]

# Verify files exist
for file_path in sources:
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")

# Define the extension
extensions = [
    Extension(
        "assemble_matrices",
        sources,
        include_dirs=[np.get_include(), current_dir],
        extra_compile_args=["/Ox"] if os.name == 'nt' else ["-O3"],
    )
]

setup(
    name="assemble_matrices",
    ext_modules=cythonize(extensions),
)