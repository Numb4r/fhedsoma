import os
import pickle
import numpy as np
import sys

# Garante o acesso ao binário C++
import ckks_python

class CKKSContext:
    """
    Gerenciador de estado criptográfico.
    Centraliza parâmetros, chaves e instâncias de operação (Encoder, Encryptor, etc.)
    """
    def __init__(self, N: int, bits: int, scale_bits: int = 40):
        self.N = N
        self.bits = bits
        self.q = ckks_python.generate_prime(bits, N)
        self.scale = 2**scale_bits

        # Inicializa o maquinário base
        keygen = ckks_python.KeyGenerator(self.N, self.q)
        self.sk = keygen.gen_secret_key()
        self.pk = keygen.gen_public_key(self.sk)

        # Componentes operacionais
        self.encoder = ckks_python.Encoder(self.N, self.q, self.scale)
        self.encryptor = ckks_python.Encryptor(self.N, self.q, self.pk)
        self.decryptor = ckks_python.Decryptor(self.N, self.q, self.sk)
        self.evaluator = ckks_python.Evaluator(self.N, self.q)

    @classmethod
    def from_keys(cls, N: int, q: int, scale: int,
                  pk_bytes: bytes, sk_bytes: bytes = None,
                  public_only: bool = False) -> "CKKSContext":
        """Reconstrói um CKKSContext a partir de chaves serializadas, sem gerar novas."""
        ctx = cls.__new__(cls)
        ctx.N = N
        ctx.q = q
        ctx.scale = scale
        ctx.bits = q.bit_length()
        ctx.pk = ckks_python.PublicKey.from_bytes(pk_bytes)
        ctx.encoder = ckks_python.Encoder(N, q, scale)
        ctx.encryptor = ckks_python.Encryptor(N, q, ctx.pk)
        ctx.evaluator = ckks_python.Evaluator(N, q)
        if public_only or sk_bytes is None:
            ctx.sk = None
            ctx.decryptor = None
        else:
            ctx.sk = ckks_python.SecretKey.from_bytes(sk_bytes)
            ctx.decryptor = ckks_python.Decryptor(N, q, ctx.sk)
        return ctx

    def save(self, path: str) -> None:
        """Salva chaves e parâmetros em disco (3 arquivos em `path/`)."""
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/params.pkl", "wb") as f:
            pickle.dump({"N": self.N, "q": self.q, "scale": self.scale}, f)
        with open(f"{path}/pk.bin", "wb") as f:
            f.write(self.pk.to_bytes())
        if self.sk is not None:
            with open(f"{path}/sk.bin", "wb") as f:
                f.write(self.sk.to_bytes())

    @classmethod
    def load(cls, path: str, public_only: bool = False) -> "CKKSContext":
        """Carrega contexto do disco. `public_only=True` para o servidor (sem SK)."""
        with open(f"{path}/params.pkl", "rb") as f:
            p = pickle.load(f)
        pk_bytes = open(f"{path}/pk.bin", "rb").read()
        sk_path = f"{path}/sk.bin"
        sk_bytes = None if public_only or not os.path.exists(sk_path) else open(sk_path, "rb").read()
        return cls.from_keys(
            N=p["N"], q=p["q"], scale=p["scale"],
            pk_bytes=pk_bytes, sk_bytes=sk_bytes,
            public_only=public_only,
        )

    def __getstate__(self):
        return {
            "N":     self.N,
            "q":     self.q,
            "scale": self.scale,
            "sk":    self.sk.to_bytes() if self.sk is not None else None,
            "pk":    self.pk.to_bytes(),
        }

    def __setstate__(self, state):
        self.N = state["N"]
        self.q = state["q"]
        self.scale = state["scale"]
        self.bits = state["q"].bit_length()
        self.pk = ckks_python.PublicKey.from_bytes(state["pk"])
        self.encoder = ckks_python.Encoder(self.N, self.q, self.scale)
        self.encryptor = ckks_python.Encryptor(self.N, self.q, self.pk)
        self.evaluator = ckks_python.Evaluator(self.N, self.q)
        if state["sk"] is not None:
            self.sk = ckks_python.SecretKey.from_bytes(state["sk"])
            self.decryptor = ckks_python.Decryptor(self.N, self.q, self.sk)
        else:
            self.sk = None
            self.decryptor = None

    def make_public(self):
        """
        Destrói o acesso à chave secreta e ao decifrador.
        Utilize este método no Servidor Flower para garantir que ele seja 'cego'
        e incapaz de decifrar o modelo.
        """
        self.sk = None
        self.decryptor = None

    def export_public_key(self) -> bytes:
        return self.pk.to_bytes()

    def import_public_key(self, pk_bytes: bytes):
        self.pk = ckks_python.PublicKey.from_bytes(pk_bytes)
        self.encryptor = ckks_python.Encryptor(self.N, self.q, self.pk)
class CKKSTensor:
    """
    Representação de um vetor cifrado de alto nível com Auto-Chunking e Metadados.
    Trafega de forma 100% transparente pela rede, preservando suas dimensões originais.
    """
    def __init__(self, context: CKKSContext, data, original_size: int = None):
        self.context = context
        self.ciphertexts = []
        self.original_size = original_size or 0
        slots = self.context.N // 2
        
        # 1. Inicialização a partir de Objetos C++ já existentes (Somas)
        if isinstance(data, list) and all(isinstance(x, ckks_python.Ciphertext) for x in data):
            self.ciphertexts = data
            
        # 2. Inicialização a partir da Rede (Deserialização do Flower)
        elif isinstance(data, list) and all(isinstance(x, bytes) for x in data):
            # LÊ O CABEÇALHO: O primeiro item da lista é o tamanho original em 8 bytes
            self.original_size = int.from_bytes(data[0], byteorder='little')
            
            # Reconstrói os cifrotextos do restante da lista
            self.ciphertexts = [ckks_python.Ciphertext.from_bytes(b) for b in data[1:]]
            
        # 3. Inicialização a partir do TensorFlow / NumPy (Cifragem em tempo real)
        else:
            array_np = np.array(data, dtype=np.float64).flatten()
            self.original_size = array_np.size
            
            # Auto-Chunking automático
            for i in range(0, self.original_size, slots):
                chunk = array_np[i : i + slots]
                pt_poly = self.context.encoder.encode(chunk)
                pt = ckks_python.Plaintext(pt_poly)
                ct = self.context.encryptor.encrypt(pt)
                self.ciphertexts.append(ct)

    def decrypt(self):
        """
        Decifra, une os chunks, remove a parte imaginária e corta o ruído de cauda.
        100% Automático. Sem necessidade de passar argumentos.
        """
        if self.context.sk is None or self.context.decryptor is None:
            raise PermissionError("Operação bloqueada: Contexto público não possui a chave secreta.")
            
        result_chunks = []
        for ct in self.ciphertexts:
            pt = self.context.decryptor.decrypt(ct)
            poly = ckks_python.Poly(self.context.N, self.context.q, pt.poly)
            decoded = self.context.encoder.decode(poly)
            result_chunks.append(np.real(decoded)) # Remove a parte imaginária ('j')
            
        # Remonta o array gigante e aplica o corte milimétrico usando o metadado
        flat_result = np.concatenate(result_chunks)
        return flat_result[:self.original_size]

    def serialize(self) -> list[bytes]:
        """
        Extrai o payload para gRPC.
        Adiciona o 'original_size' empacotado em 8 bytes no índice [0] da lista.
        """
        meta_bytes = self.original_size.to_bytes(8, byteorder='little')
        payload = [meta_bytes] + [ct.to_bytes() for ct in self.ciphertexts]
        return payload

    # ==========================================
    # Sobrecarga de Operadores 
    # ==========================================
    def __add__(self, other):
        if not isinstance(other, CKKSTensor):
            raise TypeError("A adição homomórfica requer dois CKKSTensors.")
        if len(self.ciphertexts) != len(other.ciphertexts):
            raise ValueError("Incompatibilidade de dimensões nos tensores.")
            
        result_cts = []
        for ct1, ct2 in zip(self.ciphertexts, other.ciphertexts):
            result_cts.append(self.context.evaluator.add(ct1, ct2))
            
        # Herda o tamanho original para o novo objeto agregado
        return CKKSTensor(self.context, result_cts, original_size=self.original_size)

    def __iadd__(self, other):
        if not isinstance(other, CKKSTensor):
            raise TypeError("A adição homomórfica requer dois CKKSTensors.")
            
        for ct1, ct2 in zip(self.ciphertexts, other.ciphertexts):
            self.context.evaluator.add_inplace(ct1, ct2)
            
        return self