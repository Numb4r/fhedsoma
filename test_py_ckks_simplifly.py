import numpy as np
from ckks_python_simplify import CKKSContext, CKKSTensor

ctx = CKKSContext(N=8192, bits=62, scale_bits=60)

# Um array com 10.000 elementos (MAIOR que o limite de 4096)
# pesos_locais = np.arange(2, dtype=np.float64)
pesos_locais=[1,2] 
tamanho_exato = len(pesos_locais)

# 1. Cifragem (A classe vai gerar 3 chunks silenciosamente)
tensor_cifrado_1 = CKKSTensor(ctx, pesos_locais)
tensor_cifrado_2 = CKKSTensor(ctx, pesos_locais)

# 2. Rede (Retorna uma lista contendo os 3 bytes de cada chunk)
payload_flower_1 = tensor_cifrado_1.serialize() 
payload_flower_2 = tensor_cifrado_2.serialize()

# 3. Servidor Flower soma tudo in-place
tensor_1_servidor = CKKSTensor(ctx, payload_flower_1)
tensor_2_servidor = CKKSTensor(ctx, payload_flower_2)
tensor_1_servidor += tensor_2_servidor 

payload_flower_3 = tensor_1_servidor.serialize()

# 4. Cliente decifra, passando o tamanho exato que o modelo TF espera
tensor_3_cliente = CKKSTensor(ctx, payload_flower_3)
resultado = tensor_3_cliente.decrypt()
for i in range(len(resultado)):
    print(resultado[i])  # Deve imprimir 0, 6, 12, ..., 19998
print(f"Tamanho processado perfeitamente: {len(resultado)}")
for i in range(len(resultado)):
    assert 2*pesos_locais[i] - resultado[i] <= 1e-13, f"Erro na posição {i}: esperado {2*pesos_locais[i]}, obtido {resultado[i]} : Erro numérico excede o limite de precisão! {2*pesos_locais[i] - resultado[i]} > 1e-14"
    print(f"Ruido numerico na posição {i}: {2*pesos_locais[i] - resultado[i]}")  # Imprime o ruído numérico para cada elemento
# Saída: Tamanho processado perfeitamente: 10000