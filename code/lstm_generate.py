from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Lambda, Activation
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import random
import re

# model parameters
embed_dim = 128
lstm_out = 150
batch_size = 32

# load text
with open("../data/shakespeare.txt", 'r') as f:
    text = f.read()

X = []
Y = []

# preprocessing
lines = text.split('\n')
newtxt = ""
for line in lines:
    if line == '':
        continue
    if line.strip().isdigit():
        continue
    newtxt += line + '\n'

text = newtxt

# get two dictionaries
num = len(set(text))
char2index = {}
index2char = {}
count = 0
for char in text:
    if char not in char2index:
        char2index[char] = count
        index2char[count] = char
        count += 1

# define the model
model = Sequential()
model.add(LSTM(lstm_out, input_shape=(40, num)))
model.add(Dense(num))
model.add(Lambda(lambda x: x / 1.5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/cp.ckpt",
                                                 save_weights_only=False,
                                                 verbose=1)

# generate sequence char by char
def genertate(model):
    text = "shall i compare thee to a summer's day?\n"
    for i in range(600):
        x = []
        for j in range(40):
            tmp = [0] * num
            index = char2index[text[i + j]]
            tmp[index] = 1
            x.append(tmp)
        x = np.array(x).reshape((1, len(x), len(x[1]))).astype(np.float32)
        y = model.predict(x)[0]
        index = random.choices([i for i in range(num)], y)[0]
        char = index2char[index]
        text += char
    print(text)


if __name__ == '__main__':
    model = load_model('checkpoints/model2')
    print(model.evaluate(X, Y, verbose=1))
    # model.save("models/lstm_model")
    #genertate(model)
