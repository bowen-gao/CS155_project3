from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import re

lstm_out = 100
batch_size = 32

with open("../data/shakespeare.txt", 'r') as f:
    text = f.read()

lines = text.split('\n')
newtxt = ""
for line in lines:
    if line == '':
        continue
    if line.strip().isdigit():
        continue
    newtxt += line + '\n'

text = newtxt

X = []
Y = []

num = len(set(text))
dic = {}
count = 0
for char in text:
    if char not in dic:
        dic[char] = count
        count += 1
text_list = []
print(dic)
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
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/model100",
                                                 save_weights_only=False,
                                                 verbose=1)

if __name__ == '__main__':
    print(X.shape, Y.shape)
    #model = load_model('checkpoints/model')
    model.fit(X, Y, epochs=100, callbacks=[cp_callback])
    model.save("models/lstm_model100")
