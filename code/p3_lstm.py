from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import re

embed_dim = 128
lstm_out = 150
batch_size = 32

with open("../data/shakespeare.txt", 'r') as f:
    text = f.read()


def preprocess(text):
    return text.split()


text = preprocess(text)
X = []
Y = []
for i, word in enumerate(text):
    text[i] = re.sub(r'[^\w]', '', word).lower()
num = len(set(text))
dic = {}
for i, word in enumerate(list(set(text))):
    dic[word] = i
for i, word in enumerate(text):
    tmp = [0] * num
    tmp[dic[word]] = 1
    text[i] = tmp
for i in range(0, len(text) - 40, 1):
    print(i)
    seq = text[i:i + 40]
    label = text[i + 40]
    X.append(seq)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

model = Sequential()
# model.add(Embedding(num, embed_dim, input_length=40))
model.add(LSTM(lstm_out, input_shape=(40, num)))
model.add(Dense(num, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# X nx40


if __name__ == '__main__':
    print(X.shape, Y.shape)
    model.fit(X, Y, epochs=10)
    model.save("models/lstm_model")

