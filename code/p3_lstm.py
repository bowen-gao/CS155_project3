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

new_txt = ""
for char in text:
    if not char.isdigit():
        new_txt += char
text = new_txt

X = []
Y = []

num = len(set(text))
dic = {}
count=0
for char in text:
    if char not in dic:
        dic[char] = count
        count+=1
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
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/model",
                                                 save_weights_only=False,
                                                 verbose=1)




if __name__ == '__main__':
    print(X.shape, Y.shape)
    model.fit(X, Y, epochs=100, callbacks=[cp_callback])
    model.save("models/lstm_model")
