import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import flwr as fl

# Desativa logs verbosos do TF para não poluir o terminal
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

NUM_CLIENTS = 9
NUM_ROUNDS = 5
EPOCHS_PER_ROUND = 1

# ==========================================
# 1. ARQUITETURA DO MODELO: ResNet-20
# ==========================================
def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    """Construtor padrão de camadas 2D para a ResNet"""
    conv = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                         kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization: x = layers.BatchNormalization()(x)
        if activation is not None: x = layers.Activation(activation)(x)
    else:
        if batch_normalization: x = layers.BatchNormalization()(x)
        if activation is not None: x = layers.Activation(activation)(x)
        x = conv(x)
    return x

def build_resnet20(input_shape=(32, 32, 3), num_classes=10):
    """Constrói a ResNet-20 canônica para CIFAR-10"""
    inputs = keras.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    # 3 estágios de blocos residuais
    num_filters_in = 16
    for stage in range(3):
        for res_block in range(3):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage > 0 and res_block == 0:
                strides = 2  # Reduz a dimensionalidade no primeiro bloco de novos estágios
            
            y = resnet_layer(inputs=x, num_filters=num_filters_in, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, activation=None)
            
            if stage > 0 and res_block == 0:
                # Projeção de atalho para equalizar dimensões
                x = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            
            x = keras.layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters_in *= 2

    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 2. PARTICIONAMENTO DE DADOS (CIFAR-10)
# ==========================================
def load_datasets(num_clients):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalização
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Divide os dados de treino de forma igual entre os clientes
    partition_size = len(x_train) // num_clients
    train_partitions = []
    
    for i in range(num_clients):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        train_partitions.append((x_train[start_idx:end_idx], y_train[start_idx:end_idx]))
        
    return train_partitions, (x_test, y_test)

# ==========================================
# 3. CLIENTE FLOWER (TensorFlow)
# ==========================================
class CifarClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = build_resnet20()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Treinamento local de 1 época
        history = self.model.fit(self.x_train, self.y_train, epochs=EPOCHS_PER_ROUND, batch_size=32, verbose=0)
        
        # NOTA FUTURA: É AQUI que interceptaremos get_weights(), aplicaremos a divisão (1/9)
        # faremos o flatten(), codificação em chunks e cifragem via CKKS.
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

def client_fn(cid: str) -> fl.client.Client:
    """Função fábrica chamada pelo Ray para instanciar clientes sob demanda"""
    partition_idx = int(cid)
    (x_train_cid, y_train_cid) = train_partitions[partition_idx]
    return CifarClient(x_train_cid, y_train_cid, x_test, y_test)

# ==========================================
# 4. ORQUESTRAÇÃO E EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    print(f"[*] Carregando e particionando CIFAR-10 para {NUM_CLIENTS} clientes...")
    train_partitions, (x_test, y_test) = load_datasets(NUM_CLIENTS)
    
    # Mostra um sumário do modelo na máquina primária antes de começar
    temp_model = build_resnet20()
    print(f"[*] Parâmetros da ResNet-20: {temp_model.count_params()}")
    del temp_model

    print("\n[*] Iniciando Simulação Federada...")
    
    # Estratégia Base (FedAvg). 
    # NOTA FUTURA: É AQUI que criaremos a CustomStrategy para agregar os Cifrotextos.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Garante que todos os 9 clientes participem da rodada
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    # Configura os recursos por cliente. 
    # Definir CPU muito baixa permite que o Ray rode vários no seu Inspiron.
    client_resources = {"num_cpus": 1, "num_gpus": 0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    print("\n✅ Treinamento Base Finalizado.")