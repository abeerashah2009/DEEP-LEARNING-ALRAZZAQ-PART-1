# Lab 18: Text Generation with RNNs

## Overview

This lab demonstrates character-level text generation using a Recurrent Neural Network (RNN) with an LSTM layer.

The training text is taken from *Alice's Adventures in Wonderland* by Lewis Carroll, downloaded from Project Gutenberg.

The model learns character patterns from the text and generates new characters based on a starting sequence.

## Objectives

- Understand how Recurrent Neural Networks work for text generation.
- Prepare and preprocess a text corpus.
- Build a character-level LSTM model using TensorFlow and Keras.
- Train the model to predict the next character.
- Generate text using different temperature values.
- Compare the quality and diversity of generated text.

## Technologies Used

- Python 3.12.3
- TensorFlow CPU 2.17.0
- Keras 3.15.1
- NumPy 1.26.4
- Ubuntu 24.04 LTS

## Project Structure

```text
Lab-18-Text-Generation-with-RNNs/
├── alice_raw.txt
├── alice_in_wonderland.txt
├── corpus.txt
├── load.py
├── prep.py
├── rnn.py
├── generated_output.txt
├── requirements.txt
└── README.md
```

The Python virtual environment is excluded from Git.

## Step 1: Text Preparation

Downloaded *Alice's Adventures in Wonderland* from Project Gutenberg.

The original text was cleaned by removing the Gutenberg header and footer.

The preprocessing script performed these operations:

- Converted the text to lowercase.
- Replaced line breaks with spaces.
- Selected the first 30,000 characters.
- Saved the processed text as `corpus.txt`.

**Final corpus size:** 30,000 characters.

## Step 2: Character-Level Preprocessing

The model identifies unique characters and maps them to numerical indexes.

| Parameter | Value |
|---|---:|
| Unique characters | 48 |
| Sequence length | 100 |
| Step size | 3 |
| Training sequences | 9,967 |

Each training sequence contains 100 characters, and the target is the next character.

One-hot encoding was used to represent input and target characters.

## Step 3: LSTM Model Architecture

The Sequential model contains:

| Layer | Configuration |
|---|---|
| Input | 100 time steps × 48 characters |
| LSTM | 128 units |
| Dense | 48 units |
| Activation | Softmax |
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |

**Total parameters:** 96,816

**Trainable parameters:** 96,816

## Step 4: Model Training

The model was trained using:

- Batch size: 128
- Epochs: 10
- Training sequences: 9,967

### Training Results

The first training run produced the following loss values:

| Epoch | Training Loss |
|---:|---:|
| 1 | 3.0813 |
| 2 | 2.9176 |
| 3 | 2.8435 |
| 4 | 2.7321 |
| 5 | 2.6227 |
| 6 | 2.5333 |
| 7 | 2.4570 |
| 8 | 2.3881 |
| 9 | 2.3338 |
| 10 | 2.2963 |

The training loss decreased across all ten epochs, indicating that the model learned patterns from the training corpus.

## Step 5: Text Generation

A sampling function was implemented to select the next character using predicted probabilities and a temperature value.

The model used:

- Seed length: 100 characters
- New characters generated: 200
- Total output length: 300 characters
- Temperature values: 0.2 and 0.8

Sample excerpts from the generated outputs are saved in `generated_output.txt`. The script verifies that each complete generated output contains 300 characters.

### Temperature Comparison

**Temperature 0.2 — Lower randomness**

The generated output was more predictable but showed repetitive phrases and character patterns.

**Temperature 0.8 — Higher randomness**

The generated output contained more varied character combinations but included many misspelled and meaningless words.

### Observations

- Lower temperature makes character selection more predictable.
- Higher temperature increases randomness and output diversity.
- The model learned basic character-level patterns.
- The generated text did not consistently maintain correct grammar or meaningful sentences.

## Limitations

- The training corpus was limited to 30,000 characters.
- The model was trained for only 10 epochs.
- The model predicts one character at a time.
- The generated text sometimes contains repeated patterns and meaningless words.

## Possible Improvements

- Train with a larger text corpus.
- Increase the number of training epochs.
- Experiment with different LSTM configurations.
- Compare LSTM with GRU networks.
- Explore Transformer-based text generation.

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the source text:

```bash
python load.py
python prep.py
```

Run model training and text generation:

```bash
python rnn.py
```

The script trains the model and generates samples at temperatures 0.2 and 0.8.

## Conclusion

This lab provided practical experience with character-level text generation using Recurrent Neural Networks and LSTM.

The model successfully learned character patterns from the prepared text corpus and generated new text using different temperature values.

The temperature comparison demonstrated the relationship between randomness, repetition, and diversity in generated text.

**Lab 18 implementation and training completed successfully.**
