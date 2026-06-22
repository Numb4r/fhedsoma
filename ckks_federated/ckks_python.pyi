"""
Motor Criptográfico Homomórfico (CKKS) para Aprendizado Federado.
Desenvolvido para operações de preservação de privacidade em redes neurais.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = ['Ciphertext', 'Decryptor', 'Encoder', 'Encryptor', 'Evaluator', 'KeyGenerator', 'Plaintext', 'Poly', 'PublicKey', 'SecretKey', 'generate_prime']

class Ciphertext:
    """
    Representação de um texto cifrado no esquema RLWE.
    Matematicamente composto por um par de polinômios (c0, c1) no anel R_q.
    """
    @staticmethod
    def from_bytes(data: bytes) -> Ciphertext:
        """
        Reconstrói um Cifrotexto a partir de um fluxo binário recebido via rede (ex: gRPC/Flower).
        """
        ...
    def __init__(self, c0: Poly, c1: Poly) -> None:
        """Instancia um cifrotexto diretamente a partir de dois polinômios."""
        ...
    def to_bytes(self) -> bytes:
        """
        Serializa o cifrotexto para transporte em rede.
        Gera um payload binário compactado contendo N, q e os coeficientes de c0 e c1.
        """
        ...
    @property
    def c0(self) -> numpy.typing.NDArray[numpy.uint64]:
        """Acessa o array de coeficientes do polinômio c0."""
        ...
    @property
    def c1(self) -> numpy.typing.NDArray[numpy.uint64]:
        """Acessa o array de coeficientes do polinômio c1."""
        ...

class Decryptor:
    """
    Módulo responsável por reverter a cifragem RLWE.
    Exige a Chave Secreta estrita. Nunca deve ser instanciado no Servidor Federado.
    """
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex, sk: SecretKey) -> None:
        """Inicializa o decifrador com os parâmetros do anel e a Chave Secreta."""
        ...
    def decrypt(self, ct: Ciphertext) -> Plaintext:
        """
        Aplica a chave secreta s para computar o polinômio em claro aproximado: m ≈ c0 + c1*s (mod q).
        """
        ...

class Encoder:
    """
    Responsável pela imersão de vetores reais/complexos no anel polinomial.
    Aplica o fator de escala (Delta) para preservar a precisão de ponto flutuante em inteiros uint64.
    """
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex, scale: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """Inicializa o codificador. 'scale' define a precisão das frações (ex: 2**40)."""
        ...
    def decode(self, p: Poly) -> numpy.typing.NDArray[numpy.complex128]:
        """
        Reverte um polinômio em claro para um vetor de números reais/complexos decodificados.
        """
        ...
    @typing.overload
    def encode(self, z: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> Poly:
        """
        Codifica um array de reais (float64) em um polinômio matemático. 
        Tamanho máximo do array: N/2.
        """
        ...
    @typing.overload
    def encode(self, z: typing.Annotated[numpy.typing.ArrayLike, numpy.complex128]) -> Poly:
        """Codifica um array complexo (complex128) em um polinômio matemático."""
        ...
    @property
    def degree(self) -> int:
        """Retorna o grau N do polinômio."""
        ...
    @property
    def slots(self) -> int:
        """Retorna o número máximo de elementos suportados por cifrotexto (N/2)."""
        ...

class Encryptor:
    """
    Aplica a segurança semântica inserindo ruído Gaussiano amostrado.
    Mapeia polinômios em claro para cifrotextos utilizando a Chave Pública.
    """
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex, pk: PublicKey, sigma: typing.SupportsFloat | typing.SupportsIndex = 3.2, seed: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """Inicializa o cifrador acoplado à Chave Pública do ambiente federado."""
        ...
    def encrypt(self, pt: Plaintext) -> Ciphertext:
        """
        Cifra um texto em claro. Retorna o par de polinômios ruidoso (c0, c1).
        """
        ...

class Evaluator:
    """
    Motor de álgebra homomórfica.
    Executa cálculos sobre dados cifrados sem acesso às chaves. É o único módulo matemático que deve operar no Servidor Flower.
    """
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex) -> None:
        """Inicializa o avaliador (Não requer chaves criptográficas)."""
        ...
    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """
        Soma homomórfica clássica. Aloca e retorna um novo objeto Ciphertext na memória.
        Atenção: Evite usar em laços longos para prevenir fragmentação de RAM.
        """
        ...
    def add_inplace(self, ct1: Ciphertext, ct2: Ciphertext) -> None:
        """
        Soma homomórfica otimizada em hardware. Executa ct1 += ct2 in-place.
        Mandatório para uso iterativo no 'aggregate_fit' do Servidor Federado.
        """
        ...

class KeyGenerator:
    """Instancia o par de chaves base do esquema de Aprendizado com Erros em Anéis (RLWE)."""
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex, sigma: typing.SupportsFloat | typing.SupportsIndex = 3.2, seed: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        ...
    def gen_public_key(self, sk: SecretKey) -> PublicKey:
        """Gera a chave pública pk = (a, -a*s + e) acoplada à chave secreta fornecida."""
        ...
    def gen_secret_key(self, hamming_weight: typing.SupportsInt | typing.SupportsIndex = 0) -> SecretKey:
        """Gera o polinômio secreto 's' com coeficientes ternários {-1, 0, 1}."""
        ...

class Plaintext:
    """Container envolucro para um polinômio em claro antes da cifragem ou após decifragem."""
    def __init__(self, poly: Poly) -> None:
        ...
    @property
    def poly(self) -> numpy.typing.NDArray[numpy.uint64]:
        """Acessa o buffer bruto de coeficientes uint64 do texto em claro."""
        ...

class Poly:
    """Estrutura fundamental que representa um polinômio no anel quociente R_q = Z_q[X]/(X^N+1)."""
    def __add__(self, arg0: Poly) -> Poly:
        ...
    @typing.overload
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @typing.overload
    def __init__(self, N: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsInt | typing.SupportsIndex, coeffs: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]) -> None:
        ...
    @property
    def N(self) -> int:
        ...
    @property
    def coeffs(self) -> numpy.typing.NDArray[numpy.uint64]:
        ...
    @coeffs.setter
    def coeffs(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64]) -> None:
        ...
    @property
    def q(self) -> int:
        ...

class PublicKey:
    """Chave assimétrica pública utilizada para cifrar as atualizações do modelo local (Pesos da ResNet)."""
    @staticmethod
    def from_bytes(data: bytes) -> PublicKey:
        """Reconstrói a chave pública a partir do payload recebido do servidor."""
        ...
    def to_bytes(self) -> bytes:
        """Exporta a chave pública para tráfego em rede gRPC."""
        ...
    @property
    def pk0(self) -> numpy.typing.NDArray[numpy.uint64]:
        ...
    @property
    def pk1(self) -> numpy.typing.NDArray[numpy.uint64]:
        ...

class SecretKey:
    """Chave privada rigorosamente restrita ao Cliente Local. Nunca transmite via rede."""
    @staticmethod
    def from_bytes(data: bytes) -> SecretKey:
        """Carrega a chave secreta a partir do disco local."""
        ...
    def to_bytes(self) -> bytes:
        """Exporta a chave secreta estritamente para armazenamento seguro em disco."""
        ...
    @property
    def s(self) -> numpy.typing.NDArray[numpy.uint64]:
        ...

def generate_prime(bit_size: typing.SupportsInt | typing.SupportsIndex, N: typing.SupportsInt | typing.SupportsIndex) -> int:
    """
    Busca um Primo de Proth suportado pela NTT.
    Retorna o primeiro primo q com 'bit_size' bits exatos satisfazendo estritamente a congruência q ≡ 1 (mod 2N).
    """
    ...