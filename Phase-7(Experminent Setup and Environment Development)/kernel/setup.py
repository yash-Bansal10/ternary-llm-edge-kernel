from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='ternary_edge',
    ext_modules=[
        CppExtension(
            name='ternary_edge',
            sources=['poc_ternary_mac.cpp'],
            extra_compile_args=['-O3', '-march=native'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
