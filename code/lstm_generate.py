from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model
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
new_txt = ""
for char in text:
    if not char.isdigit():
        new_txt += char
text = new_txt

num = len(set(text))
char2index = {}
index2char = {}
count = 0
for char in text:
    if char not in char2index:
        char2index[char] = count
        index2char[count] = char
        count += 1
print(char2index)
text_list = []
for char in text:
    tmp = [0] * num
    tmp[char2index[char]] = 1
    text_list.append(tmp)
for i in range(0, len(text) - 40, 1):
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
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/cp.ckpt",
                                                 save_weights_only=False,
                                                 verbose=1)


def genertate(model):
    text = "shall i compare thee to a summer's day?\n"
    for i in range(100):
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
    model = load_model('checkpoints/model')
    #print(newmodel.evaluate(X, Y, verbose=1))
    # model.save("models/lstm_model")
    genertate(model)
