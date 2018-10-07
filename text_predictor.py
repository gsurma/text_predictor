import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# I/O
if len(sys.argv) < 2:
    dataset = "shakespeare"
else:
    dataset = sys.argv[1]
print "Selected dataset: " + str(dataset)
dir = "./data/" + dataset
input_data = open(dir + "/input.txt", 'r').read()
output_data = open(dir + "/output.txt", "w")
chars = list(set(input_data))
data_size, vocabulary_size = len(input_data), len(chars)
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

# Hyperparameters
HIDDEN_LAYER_SIZE = 100
SEQUENCE_LENGTH = 25
LEARNING_RATE = 1e-1
ADAGRAD_UPDATE_RATE = 1e-8
LOGGING_FREQUENCY = 1e3
TEXT_SAMPLING_FREQUENCY = 1e3
GRADIENT_LIMIT = 5

# # Model
# input_to_hidden = np.random.randn(HIDDEN_LAYER_SIZE, vocabulary_size) * 0.01
# hidden_to_hidden = np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * 0.01
# hidden_to_output = np.random.randn(vocabulary_size, HIDDEN_LAYER_SIZE) * 0.01
# hidden_bias = np.zeros((HIDDEN_LAYER_SIZE, 1))
# output_bias = np.zeros((vocabulary_size, 1))

class RNNModel:

    def __init__(self, random_init=False):
        self.input_to_hidden = (np.random.randn(HIDDEN_LAYER_SIZE, vocabulary_size) * 0.01) if random_init else np.zeros((HIDDEN_LAYER_SIZE, vocabulary_size))
        self.hidden_to_hidden = (np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * 0.01) if random_init else np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
        self.hidden_to_output = (np.random.randn(vocabulary_size, HIDDEN_LAYER_SIZE) * 0.01) if random_init else np.zeros((vocabulary_size, HIDDEN_LAYER_SIZE))
        self.hidden_bias = np.zeros((HIDDEN_LAYER_SIZE, 1))
        self.output_bias = np.zeros((vocabulary_size, 1))

    def __iter__(self):
        return iter([self.input_to_hidden, self.hidden_to_hidden, self.hidden_to_output, self.hidden_bias, self.output_bias])


base = RNNModel(random_init=True)


def update(inputs, targets, hidden_previous):
    input_state, hidden_state, output_state, probabilites_state = {}, {}, {}, {}
    hidden_state[-1] = np.copy(hidden_previous)
    loss = 0
    # forward pass
    for t in xrange(len(inputs)):
        input_state[t] = np.zeros((vocabulary_size, 1))  # encode in 1-of-k representation
        input_state[t][inputs[t]] = 1
        hidden_state[t] = np.tanh(np.dot(base.input_to_hidden, input_state[t]) + np.dot(base.hidden_to_hidden, hidden_state[t - 1]) + base.hidden_bias)  # hidden state
        output_state[t] = np.dot(base.hidden_to_output, hidden_state[t]) + base.output_bias  # unnormalized log probabilities for next chars
        probabilites_state[t] = np.exp(output_state[t]) / np.sum(np.exp(output_state[t]))  # probabilities for next chars
        loss += -np.log(probabilites_state[t][targets[t], 0])  # softmax (cross-entropy loss)
    # backward pass: compute gradients going backwards
    gradient_input_to_hidden, gradient_hidden_to_hidden, gradient_hidden_to_output = np.zeros_like(base.input_to_hidden), np.zeros_like(base.hidden_to_hidden), np.zeros_like(base.hidden_to_output)
    gradient_hidden_bias, gradient_output_bias = np.zeros_like(base.hidden_bias), np.zeros_like(base.output_bias)
    gradient_hidden_next = np.zeros_like(hidden_state[0])
    for t in reversed(xrange(len(inputs))):
        gradient_output = np.copy(probabilites_state[t])
        gradient_output[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        gradient_hidden_to_output += np.dot(gradient_output, hidden_state[t].T)
        gradient_output_bias += gradient_output
        gradient_hidden = np.dot(base.hidden_to_output.T, gradient_output) + gradient_hidden_next  # backprop into h
        gradient_hidden_raw = (1 - hidden_state[t] * hidden_state[t]) * gradient_hidden  # backprop through tanh nonlinearity
        gradient_hidden_bias += gradient_hidden_raw
        gradient_input_to_hidden += np.dot(gradient_hidden_raw, input_state[t].T)
        gradient_hidden_to_hidden += np.dot(gradient_hidden_raw, hidden_state[t - 1].T)
        gradient_hidden_next = np.dot(base.hidden_to_hidden.T, gradient_hidden_raw)
    for gradient_param in [gradient_input_to_hidden, gradient_hidden_to_hidden, gradient_hidden_to_output, gradient_hidden_bias, gradient_output_bias]:
        np.clip(gradient_param, -GRADIENT_LIMIT, GRADIENT_LIMIT, out=gradient_param)  # clip to mitigate exploding gradients
    return loss, gradient_input_to_hidden, gradient_hidden_to_hidden, gradient_hidden_to_output, gradient_hidden_bias, gradient_output_bias, hidden_state[len(inputs) - 1]


def plot_smooth_loss(smooth_losses):
    plt.plot(range(len(smooth_losses)), smooth_losses)
    plt.title(dataset)
    plt.xlabel('iteration')
    plt.ylabel('smooth loss')
    plt.savefig(dir + "/loss.png", bbox_inches="tight")
    plt.close()


def sample_text(previous_hidden, seed_index, n):
    input = np.zeros((vocabulary_size, 1))
    input[seed_index] = 1
    indices = []
    for t in xrange(n):
        previous_hidden = np.tanh(np.dot(base.input_to_hidden, input) + np.dot(base.hidden_to_hidden, previous_hidden) + base.hidden_bias)
        output = np.dot(base.hidden_to_output, previous_hidden) + base.output_bias
        probability = np.exp(output) / np.sum(np.exp(output))
        index = np.random.choice(range(vocabulary_size), p=probability.ravel())
        input = np.zeros((vocabulary_size, 1))
        input[index] = 1
        indices.append(index)
    return indices


def rnn():
    iteration = 0
    data_pointer = 0

    memory_input_to_hidden = np.zeros_like(base.input_to_hidden)
    memory_hidden_to_hidden = np.zeros_like(base.hidden_to_hidden)
    memory_hidden_to_output = np.zeros_like(base.hidden_to_output)
    memory_bias_hidden = np.zeros_like(base.hidden_bias)
    memory_bias_output = np.zeros_like(base.output_bias)

    memory = RNNModel()

    smooth_loss = None
    smooth_losses = []
    resets = 0
    while True:
        if data_pointer + SEQUENCE_LENGTH + 1 >= len(input_data) or iteration == 0:
            resets += 1
            hidden_previous = np.zeros((HIDDEN_LAYER_SIZE, 1))
            data_pointer = 0
            print('{{"metric": "reset", "value": {}}}'.format(resets))
        inputs = [char_to_index[ch] for ch in input_data[data_pointer:data_pointer + SEQUENCE_LENGTH]]
        targets = [char_to_index[ch] for ch in input_data[data_pointer + 1:data_pointer + SEQUENCE_LENGTH + 1]]

        if iteration % TEXT_SAMPLING_FREQUENCY == 0:
            sample_index = sample_text(hidden_previous, inputs[0], 200)
            txt = ''.join(index_to_char[index] for index in sample_index)
            output_data.write("Iteration: " + str(iteration) + "\n")
            output_data.write(txt + "\n")
            output_data.write("\n")

        # forward seq_length characters through the net and fetch gradient
        loss, gradient_input_to_hidden, gradient_hidden_to_hidden, gradient_hidden_to_output, gradient_hidden_bias, gradient_output_bias, hidden_previous = update(inputs, targets, hidden_previous)

        smooth_loss = loss if smooth_loss is None else smooth_loss * 0.999 + loss * 0.001

        smooth_losses.append(smooth_loss)
        if iteration % LOGGING_FREQUENCY == 0:
            print('{{"metric": "iteration", "value": {}}}'.format(iteration))
            print('{{"metric": "smooth_loss", "value": {}}}'.format(smooth_loss))
            print('{{"metric": "loss", "value": {}}}'.format(loss))
            plot_smooth_loss(smooth_losses)

        # perform parameter update with Adagrad
        for base_param, gradient_param, memory_param in zip(base,
                                                            [gradient_input_to_hidden, gradient_hidden_to_hidden, gradient_hidden_to_output, gradient_hidden_bias, gradient_output_bias],
                                                            memory):
            memory_param += gradient_param * gradient_param
            base_param += -LEARNING_RATE * gradient_param / np.sqrt(memory_param + ADAGRAD_UPDATE_RATE)

        data_pointer += SEQUENCE_LENGTH
        iteration += 1


if __name__ == "__main__":
    rnn()
