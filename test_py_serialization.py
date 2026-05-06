import numpy as np
import sys
import os

sys.path.insert(0, "build")
import ckks_python

def main():
    print("=== Suíte de Testes Abrangente: Motor CKKS ===\n")

    # ==========================================
    # 1. Geração de Parâmetros e Chaves
    # ==========================================
    N = 8192
    bits = 60
    q = ckks_python.generate_prime(bits, N)
    slots = N // 2
    
    print("[1] Geração de Parâmetros")
    print(f"    N: {N}, q: {q}")
    
    keygen = ckks_python.KeyGenerator(N, q)
    sk = keygen.gen_secret_key()
    pk = keygen.gen_public_key(sk)

    # ==========================================
    # 2. Persistência (Serialização de Chaves)
    # ==========================================
    print("\n[2] Teste de I/O em Disco (Persistência)")
    
    with open("secret_key.bin", "wb") as f:
        f.write(sk.to_bytes())
    with open("secret_key.bin", "rb") as f:
        sk = ckks_python.SecretKey.from_bytes(f.read())
        
    with open("public_key.bin", "wb") as f:
        f.write(pk.to_bytes())
    with open("public_key.bin", "rb") as f:
        pk = ckks_python.PublicKey.from_bytes(f.read())
        
    print("    ✅ Chaves serializadas e carregadas do disco com sucesso.")

    # ==========================================
    # 3. Codificação e Cifragem
    # ==========================================
    print("\n[3] Teste de Codificação e Cifragem")
    
    weights_A = np.random.uniform(-1.0, 1.0, slots)
    weights_B = np.random.uniform(-1.0, 1.0, slots)

    encoder = ckks_python.Encoder(N, q, 2**40)
    pt_A = ckks_python.Plaintext(encoder.encode(weights_A))
    pt_B = ckks_python.Plaintext(encoder.encode(weights_B))

    encryptor = ckks_python.Encryptor(N, q, pk)
    ct_A = encryptor.encrypt(pt_A)
    ct_B = encryptor.encrypt(pt_B)
    print("    ✅ Vetores A e B codificados e cifrados.")

    # ==========================================
    # 4. Serialização de Rede (Cifrotexto)
    # ==========================================
    print("\n[4] Teste de Tráfego de Rede (Cifrotexto)")
    
    ct_A_bytes = ct_A.to_bytes()
    ct_A_recuperado = ckks_python.Ciphertext.from_bytes(ct_A_bytes)
    tamanho_kb = len(ct_A_bytes) / 1024
    print(f"    ✅ Cifrotexto A serializado ({tamanho_kb:.2f} KB) e recuperado.")

    # ==========================================
    # 5. Avaliação Homomórfica
    # ==========================================
    print("\n[5] Teste de Adição Homomórfica")
    evaluator = ckks_python.Evaluator(N, q)

    # Teste 5.1: Adição alocando novo objeto na memória
    ct_soma_novo = evaluator.add(ct_A_recuperado, ct_B)
    print("    ✅ evaluator.add() executado com sucesso.")

    # Teste 5.2: Adição in-place (Mandatório para a RAM do servidor no FL)
    evaluator.add_inplace(ct_A_recuperado, ct_B)
    print("    ✅ evaluator.add_inplace() executado com sucesso.")

    # ==========================================
    # 6. Decifragem e Aferição de Erro Crítico
    # ==========================================
    print("\n[6] Validação Matemática da Decifragem")
    decryptor = ckks_python.Decryptor(N, q, sk)

    # Decifragem do resultado 'add'
    pt_soma_novo = decryptor.decrypt(ct_soma_novo)
    poly_soma_novo = ckks_python.Poly(N, q, pt_soma_novo.poly)
    pesos_soma_novo = encoder.decode(poly_soma_novo)

    # Decifragem do resultado 'add_inplace'
    pt_inplace = decryptor.decrypt(ct_A_recuperado)
    poly_inplace = ckks_python.Poly(N, q, pt_inplace.poly)
    pesos_inplace = encoder.decode(poly_inplace)

    # Cálculo do erro em relação à soma teórica em texto claro
    esperado = weights_A + weights_B
    erro_add = np.max(np.abs(esperado - np.real(pesos_soma_novo[:slots])))
    erro_inplace = np.max(np.abs(esperado - np.real(pesos_inplace[:slots])))

    print(f"    Erro máximo (add):         {erro_add:.5e}")
    print(f"    Erro máximo (add_inplace): {erro_inplace:.5e}")

    assert erro_add < 1e-4, "FALHA CRÍTICA: A operação 'add' destruiu os dados."
    assert erro_inplace < 1e-4, "FALHA CRÍTICA: A operação 'add_inplace' corrompeu a memória."

    print("\n🏆 SUÍTE APROVADA: Todas as operações criptográficas estão matematicamente corretas.")

if __name__ == "__main__":
    main()