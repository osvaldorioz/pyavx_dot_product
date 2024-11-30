from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "avx_dot_product",  # Nombre del módulo
        ["avx_dot_product.cpp"],  # Archivo fuente
        extra_compile_args=["-mavx"],  # Activar soporte AVX
    ),
]

setup(
    name="avx_dot_product",
    version="1.0",
    author="Osvaldo R",
    description="Producto punto vectorizado con AVX y Pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
