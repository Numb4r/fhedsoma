# O uso da sua biblioteca agora é reduzido a 4 linhas:
import numpy as np
from ckks_python_simplify import CKKSContext, CKKSTensor

# 1. Inicia o contexto no cliente
ctx = CKKSContext(N=8192, bits=60, scale_bits=40)

# 2. Transforma os pesos da ResNet em um Tensor Cifrado
# pesos_locais = np.random.uniform(-1, 1, 4096)
pesos_locais = [1,2,3]
tensor_cifrado_1 = CKKSTensor(ctx, pesos_locais)
tensor_cifrado_2 = CKKSTensor(ctx, pesos_locais) # Simula outro cliente com pesos diferentes
# 3. Extrai bytes para enviar pela rede ao Flower
payload_flower_1 = tensor_cifrado_1.serialize() 
payload_flower_2 = tensor_cifrado_2.serialize()

# 4. No Servidor, reconstrói os bytes e soma in-place de forma nativa
tensor_1_servidor = CKKSTensor(ctx, payload_flower_1)
tensor_2_servidor = CKKSTensor(ctx, payload_flower_2)
tensor_3 = tensor_1_servidor + tensor_2_servidor
payload_flower_3 = tensor_3.serialize() # Envia de volta para o cliente

# No cliente, decifra o resultado
tensor_3_cliente = CKKSTensor(ctx, payload_flower_3)
resultado = tensor_3_cliente.decrypt()
print("Resultado da soma homomórfica:", resultado)
for i in range(len(resultado)):
    print(f"Slot {i}: {resultado[i]:.6f} (Esperado: {(pesos_locais[i] + pesos_locais[i]):.6f})")