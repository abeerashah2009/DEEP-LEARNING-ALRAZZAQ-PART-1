
# Lab 37: Introduction to Sequence-to-Sequence Models

## 1. Lab Overview

In this lab, I learned how Sequence-to-Sequence (Seq2Seq) models work.

I built an Encoder-Decoder model using PyTorch and trained it to reverse a sequence of five random digits.

## 2. Objectives

- Understand Sequence-to-Sequence models.
- Build an Encoder using GRU.
- Build a Decoder using GRU.
- Train the model using a small dataset.
- Evaluate model predictions.
- Understand the role of SOS and EOS tokens.

## 3. Environment

- Operating System: Ubuntu 24.04 LTS
- Programming Language: Python 3.12.3
- Deep Learning Framework: PyTorch 2.5.1+cpu
- Environment: Python Virtual Environment
- Model: GRU-based Encoder-Decoder

## 4. Model Architecture

### Encoder

The Encoder reads the input sequence one digit at a time.

Components:
- Embedding layer
- GRU layer
- Hidden state

The final hidden state is passed to the Decoder as context.

### Decoder

The Decoder generates the reversed sequence.

Components:
- Embedding layer
- ReLU activation
- GRU layer
- Linear layer
- LogSoftmax

### Special Tokens

- SOS = Start of Sequence (0)
- EOS = End of Sequence (1)
- Digits 0–9 are mapped to token IDs 2–11.

## 5. Dataset

Input example:

[6, 6, 0, 4, 8]

Expected output:

[8, 4, 0, 6, 6, EOS]

Dataset details:
- Input sequence length: 5 digits
- Output: Reversed digits followed by EOS
- Test dataset: 100 unique sequences
- Test sequences are excluded from training.

## 6. Training Configuration

- Training iterations: 6000
- Hidden size: 128
- Learning rate: 0.003
- Optimizer: Adam
- Loss function: NLLLoss
- Teacher forcing probability: 50%
- Training progress printed every 500 iterations.

## 7. Training Results

| Iteration | Average Loss |
|---|---:|
| 500 | 1.3005 |
| 1000 | 0.8898 |
| 1500 | 0.7630 |
| 2000 | 0.6928 |
| 2500 | 0.6887 |
| 3000 | 0.6416 |
| 3500 | 0.5909 |
| 4000 | 0.5650 |
| 4500 | 0.5061 |
| 5000 | 0.4792 |
| 5500 | 0.4314 |
| 6000 | 0.4258 |

## 8. Evaluation Results

The trained model was tested on 100 unseen sequences.

Final results:

- Exact Match Accuracy: 24.00%
- Correct Predictions: 24/100
- Total Test Sequences: 100

The model successfully learned some sequence-reversal patterns, but it did not predict every sequence correctly.

## 9. How to Run

Activate the virtual environment:

```bash
source ~/mlenv/bin/activate
```

Go to the lab directory:

```bash
cd ~/DEEP-LEARNING-ALRAZZAQ-PART-1/Lab-37-Introduction-to-Sequence-to-Sequence-Models
```

Check Python syntax:

```bash
python -m py_compile seq2seq.py
```

Run the model:

```bash
python seq2seq.py
```

## 10. Key Learning Outcomes

- Learned the basic concept of Seq2Seq models.
- Understood Encoder and Decoder architecture.
- Practiced building GRU networks in PyTorch.
- Learned how teacher forcing works.
- Understood SOS and EOS tokens.
- Practiced model training and evaluation.
- Learned how to measure exact-match accuracy.

## 11. Conclusion

In this lab, I implemented and trained a basic GRU-based Sequence-to-Sequence model using PyTorch.

The model learned to reverse digit sequences. The training loss decreased during training, and the final evaluation accuracy was 24%.

This practical exercise helped me understand the basic working process of Encoder-Decoder models.
