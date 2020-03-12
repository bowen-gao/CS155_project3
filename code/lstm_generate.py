from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf
import random
import re

embed_dim = 128
lstm_out = 150
batch_size = 32

with open("../data/shakespeare.txt", 'r') as f:
    text = f.read()

X = []
Y = []

num = len(set(text))
char2index = {}
index2char = {}
for i, char in enumerate(list(set(text))):
    char2index[char] = i
    index2char[i] = char

model = Sequential()
# model.add(Embedding(num, embed_dim, input_length=40))
model.add(LSTM(lstm_out, input_shape=(40, num)))
model.add(Dense(num, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# X nx40
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)


def genertate(model):
    text = "shall i compare thee to a summer's day?\n"
    for i in range(10):
        x = []
        for j in range(40):
            tmp = [0] * num
            index = char2index[text[i + j]]
            tmp[index] = 1
            x.append(tmp)
        x = np.array(x).reshape((1, len(x), len(x[1])))
        y = model.predict(x)
        index = random.choices([i for i in range(num)], y[0])[0]
        char = index2char[index]
        text += char
    print(text)


if __name__ == '__main__':
    model.load_weights("checkpoints/cp.ckpt")
    # print(model.evaluate(X, Y, verbose=1))
    # model.save("models/lstm_model")
    genertate(model)
