import glob
from setuptools import setup, find_packages

# O pybind11 fornece ajudantes nativos para o setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Varre o seu diretório em busca de todos os códigos C++
cpp_files = glob.glob("src/**/*.cpp", recursive=True) + ["bindings/module.cpp"]

# Configura o módulo de BAIXO NÍVEL em C++
ext_modules = [
    Pybind11Extension(
        "ckks_python",                     # O nome do módulo compilado (.so)
        cpp_files,                         # Os arquivos fonte
        include_dirs=["include"],          # Os arquivos de cabeçalho (.h)
        cxx_std=17,                        # Padrão rigoroso C++17
        extra_compile_args=["-O3", "-march=native", "-Wall"], # Otimização extrema no hardware de destino
    ),
]

setup(
    name="ckks_federated",
    version="1.0.0",
    license="MIT",
    classifiers=["License :: OSI Approved :: MIT License"],
    author="Yuri - UNICAMP",
    description="Motor CKKS de Alta Performance para Aprendizado Federado",
    packages=["ckks_federated"],                  # O pacote de ALTO NÍVEL
    package_data={"ckks_federated": ["*.pyi"]},   # Inclui a documentação Pylance
    ext_modules=ext_modules,                      # Acopla a lib de baixo nível
    cmdclass={"build_ext": build_ext},            # Injeta o compilador do pybind11
    install_requires=["numpy", "flwr", "tensorflow"], # Dependências automáticas
)