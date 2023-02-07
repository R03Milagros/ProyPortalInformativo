# ============= IMPORTANDO LIBRERIAS ================
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

import unicodedata

def eliminar_tildes(cadena):
    sin_tildes = ''.join((c for c in unicodedata.normalize('NFD', cadena)
                         if unicodedata.category(c) != 'Mn'))
    return sin_tildes

# leemos nuestro archivo Json en modo de escritura 
with open('intents.json', 'r') as f:
    intents = json.load(f)
# aqui recopilaremos el inverso de todas los tokens de las palabras
all_words = []
# aqui recopilaremos todos los patrones posibles 
tags = []
# vectores de palabras
xy = []
# recorrer cada oración en nuestros patrones de intenciones
for intent in intents['intents']:
    tag = intent['tag']
    # agregar a la lista de etiquetas
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenizar cada palabra en la oración
        w = tokenize(pattern)
        # añadir a nuestra lista de palabras
        all_words.extend(w)
        # agregar al par xy
        xy.append((w, tag))
# Signos de puntuacion


ignore_words = ['?', '.', '!']
# lematizamos y convertimos a minusculas  y eliminamos signos de 
# puntuacion de cada palabra
all_words = [stem(w)for w in all_words if w not in ignore_words]
all_words = [eliminar_tildes(w) for w in all_words]
print(all_words)
# eliminar duplicados y ordenar
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Imprimimos los vectores, las etiquetas y las palabras
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# crear datos de entrenamiento
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bolsa de palabras (BoW) para cada patrón_frase
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss solo necesita etiquetas de clase, no one-hot
    label = tags.index(tag)
    y_train.append(label)
# Entrenamos el modelo con los datos de entrenamiento
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hiper-parámetros
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

# Clase de los datos de entrenamiento 
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # admitir la indexación de modo que el conjunto de datos [i] se pueda 
    # usar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # podemos llamar a len (conjunto de datos) para devolver el tamaño
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entrenar a la modelo
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Pase adelantado
        outputs = model(words)
        # si y sería uno de los mejores, debemos aplicar
        # etiquetas = torch.max(etiquetas, 1)[1]
        loss = criterion(outputs, labels)
        
        # Retroceder y optimizar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
