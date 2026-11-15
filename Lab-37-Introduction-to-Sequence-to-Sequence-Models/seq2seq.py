import random
import torch
import torch.nn as nn

# Set random seeds for reproducible results
random.seed(0)
torch.manual_seed(0)

# Special tokens
SOS_token = 0
EOS_token = 1

# Model configuration
VOCAB_SIZE = 12
SEQ_LEN = 5
MAX_LENGTH = SEQ_LEN + 1
HIDDEN = 128
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_token, hidden):
        output = self.embedding(input_token).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
def generate_sequence():
    return [random.randint(0, 9) for _ in range(SEQ_LEN)]


def sequence_to_tensor(sequence):
    return torch.tensor(
        [digit + 2 for digit in sequence],
        dtype=torch.long
    ).view(-1, 1)


def target_to_tensor(sequence):
    reversed_sequence = list(reversed(sequence))
    return torch.tensor(
        [digit + 2 for digit in reversed_sequence] + [EOS_token],
        dtype=torch.long
    ).view(-1, 1)


# Create 100 unique test sequences
test_data = []

while len(test_data) < 100:
    sequence = generate_sequence()
    if sequence not in test_data:
        test_data.append(sequence)


def sample_train():
    while True:
        sequence = generate_sequence()
        if sequence not in test_data:
            return sequence
def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        _, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden
        )

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < 0.5

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            loss += criterion(
                decoder_output, target_tensor[di]
            )
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()

            loss += criterion(
                decoder_output, target_tensor[di]
            )


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
def main():
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN)
    decoder = DecoderRNN(HIDDEN, VOCAB_SIZE)

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=0.003
    )
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(), lr=0.003
    )

    criterion = nn.NLLLoss()

    N_ITERS = 6000
    print_every = 500
    print_loss_total = 0

    for iteration in range(1, N_ITERS + 1):
        training_sequence = sample_train()

        input_tensor = sequence_to_tensor(training_sequence)
        target_tensor = target_to_tensor(training_sequence)

        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion
        )

        print_loss_total += loss

        if iteration % print_every == 0:
            average_loss = print_loss_total / print_every
            print(
                f"Iteration {iteration}/{N_ITERS} "
                f"Average Loss: {average_loss:.4f}"
            )
            print_loss_total = 0
    return encoder, decoder

def evaluate(sequence, encoder, decoder):
    with torch.no_grad():
        input_tensor = sequence_to_tensor(sequence)

        encoder_hidden = encoder.initHidden()

        for ei in range(input_tensor.size(0)):
            _, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden
            )

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )

            topv, topi = decoder_output.topk(1)
            token = topi.item()

            if token == EOS_token:
                decoded_words.append("<EOS>")
                break

            decoded_words.append(str(token - 2))
            decoder_input = topi.detach()

        return decoded_words

if __name__ == "__main__":
    encoder, decoder = main()

    print("\nEvaluation examples:")

    for sequence in test_data[:5]:
        expected = [str(d) for d in reversed(sequence)] + ["<EOS>"]
        predicted = evaluate(sequence, encoder, decoder)

        print(
            f"Input: {sequence} | "
            f"Expected: {expected} | "
            f"Predicted: {predicted}"
        )

    correct = 0

    for sequence in test_data:
        expected = [str(d) for d in reversed(sequence)] + ["<EOS>"]
        predicted = evaluate(sequence, encoder, decoder)

        if predicted == expected:
            correct += 1

    accuracy = (correct / len(test_data)) * 100

    print(f"\nExact Match Accuracy: {accuracy:.2f}%")
    print(f"Correct Predictions: {correct}/{len(test_data)}")
