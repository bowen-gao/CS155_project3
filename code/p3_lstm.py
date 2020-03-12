from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf

import re

embed_dim = 128
lstm_out = 150
batch_size = 32

with open("../data/shakespeare.txt", 'r') as f:
    text = f.read()

X = []
Y = []

num = len(set(text))
dic = {}
for i, char in enumerate(list(set(text))):
    dic[char] = i
text_list = []
for char in text:
    tmp = [0] * num
    tmp[dic[char]] = 1
    text_list.append(tmp)
for i in range(0, len(text) - 40, 1):
    print(i)
    seq = text_list[i:i + 40]
    label = text_list[i + 40]
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
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/model.h5",
                                                 save_weights_only=True,
                                                 verbose=1)


def genertate(model):
    pass


if __name__ == '__main__':
    print(X.shape, Y.shape)
    model.fit(X, Y, epochs=100, callbacks=[cp_callback])
    model.save("models/lstm_model")
