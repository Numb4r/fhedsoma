import numpy as np
import sys

# Garante o acesso ao binário C++
sys.path.insert(0, "build")
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
    Representação de um vetor cifrado de alto nível.
    Sobrescreve operadores matemáticos nativos do Python para usabilidade transparente.
    """
    def __init__(self, context: CKKSContext, data):
        self.context = context
        
        if isinstance(data, ckks_python.Ciphertext):
            # Inicialização a partir de um objeto C++ existente
            self.ciphertext = data
        elif isinstance(data, bytes):
            # Inicialização a partir de tráfego de rede (Deserialização)
            self.ciphertext = ckks_python.Ciphertext.from_bytes(data)
        else:
            # Inicialização a partir de Numpy Array / Lista (Cifragem em tempo real)
            slots = self.context.N // 2
            array_np = np.array(data, dtype=np.float64)
            
            if len(array_np) > slots:
                raise ValueError(f"O vetor excede a capacidade do polinômio (Máx: {slots})")
                
            pt_poly = self.context.encoder.encode(array_np)
            pt = ckks_python.Plaintext(pt_poly)
            self.ciphertext = self.context.encryptor.encrypt(pt)

    def decrypt(self):
        if self.context.sk is None or self.context.decryptor is None:
            raise PermissionError("Operação bloqueada: Este contexto é público e não possui a chave secreta.")
            
        pt = self.context.decryptor.decrypt(self.ciphertext)
        poly = ckks_python.Poly(self.context.N, self.context.q, pt.poly)
        return self.context.encoder.decode(poly)

    def serialize(self) -> bytes:
        """Extrai o payload binário para transmissão via gRPC"""
        return self.ciphertext.to_bytes()

    # ==========================================
    # Sobrecarga de Operadores (Magia do TenSEAL)
    # ==========================================
    def __add__(self, other):
        """Suporta a sintaxe: novo_tensor = tensor1 + tensor2"""
        if not isinstance(other, CKKSTensor):
            raise TypeError("A adição homomórfica só é suportada entre CKKSTensors.")
        
        ct_result = self.context.evaluator.add(self.ciphertext, other.ciphertext)
        return CKKSTensor(self.context, ct_result)

    def __iadd__(self, other):
        """Suporta a sintaxe: tensor1 += tensor2 (Alta Performance/In-place)"""
        if not isinstance(other, CKKSTensor):
            raise TypeError("A adição homomórfica só é suportada entre CKKSTensors.")
            
        self.context.evaluator.add_inplace(self.ciphertext, other.ciphertext)
        return self