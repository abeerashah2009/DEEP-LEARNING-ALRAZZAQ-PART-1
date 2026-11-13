import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM

np.random.seed(42)
tf.random.set_seed(42)

text = open('corpus.txt', encoding='utf-8').read()

chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

print(f'Unique characters: {len(chars)}')

seq_length = 100
step = 3

sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print(f'Number of sequences: {len(sequences)}')

model = Sequential([
    keras.Input(shape=(seq_length, len(chars))),
    LSTM(128),
    Dense(len(chars), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

model.summary()

X = np.zeros((len(sequences), seq_length, len(chars)), dtype=bool)
y = np.zeros((len(sequences), len(chars)), dtype=bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

print("X shape:", X.shape)
print("y shape:", y.shape)

model.fit(X, y, batch_size=128, epochs=10)

def sample(preds, temperature=0.5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-9) / temperature
    exp = np.exp(preds)
    preds = exp / np.sum(exp)
    return np.random.choice(len(preds), p=preds)


def generate_text(seed_text, num_chars, temperature=0.5):
    assert len(seed_text) == seq_length

    input_text = seed_text
    generated_text = seed_text

    for _ in range(num_chars):
        x_pred = np.zeros((1, seq_length, len(chars)))

        for t, char in enumerate(input_text):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_char = index_to_char[sample(preds, temperature)]

        generated_text += next_char
        input_text = input_text[1:] + next_char

    return generated_text


seed_text = text[:seq_length]

with open('generated_output.txt', 'w', encoding='utf-8') as f:
    for temp in (0.2, 0.8):
        out = generate_text(seed_text, 200, temp)
        assert len(out) == len(seed_text) + 200

        result = f'\n--- temperature {temp} (length {len(out)}) ---\n{out}\n'
        print(result)
        f.write(result)
