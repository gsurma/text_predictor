import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Args
if len(sys.argv) != 2:
    print "Please select a dataset."
    print "Usage: python text_predictor.py <dataset>"
    print "Available datasets: shakespeare, wikipedia, reuters, hackernews, wikipedia, war_and_peace"
    exit(1)
else:
    dataset = sys.argv[1]
print "Selected dataset: " + str(dataset)


# I/O
dir = "./data/" + dataset
input_dir = dir + "/input.txt"
input_file = open(input_dir, "r")
input_data = input_file.read()
input_file.close()
chars = list(set(input_data))
data_size = len(input_data)
vocabulary_size = len(chars)
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for index, char in enumerate(chars)}
output_dir = dir + "/output.txt"
output_file = open(output_dir, "w")
output_file.close()


# Hyperparameters
HIDDEN_LAYER_SIZE = 512
SEQUENCE_LENGTH = 20
LEARNING_RATE = 1e-1
ADAGRAD_UPDATE_RATE = 1e-8
LOGGING_FREQUENCY = 1e3
TEXT_SAMPLING_FREQUENCY = 1e4
GRADIENT_LIMIT = 5
RANDOM_WEIGHT_INIT_FACTOR = 1e-2
LOSS_SMOOTHING_FACTOR = 0.999
TEXT_SAMPLE_LENGTH = 200


class RNNModel:

    def __init__(self, random_init=False):
        self.input_to_hidden = (np.random.randn(HIDDEN_LAYER_SIZE, vocabulary_size) * RANDOM_WEIGHT_INIT_FACTOR) if random_init else np.zeros((HIDDEN_LAYER_SIZE, vocabulary_size))
        self.hidden_to_hidden = (np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * RANDOM_WEIGHT_INIT_FACTOR) if random_init else np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
        self.hidden_to_output = (np.random.randn(vocabulary_size, HIDDEN_LAYER_SIZE) * RANDOM_WEIGHT_INIT_FACTOR) if random_init else np.zeros((vocabulary_size, HIDDEN_LAYER_SIZE))
        self.hidden_bias = np.zeros((HIDDEN_LAYER_SIZE, 1))
        self.output_bias = np.zeros((vocabulary_size, 1))

    def __iter__(self):
        return iter([self.input_to_hidden, self.hidden_to_hidden, self.hidden_to_output, self.hidden_bias, self.output_bias])


base = RNNModel(random_init=True)


def update(inputs, targets, hidden_previous):
    input_state, hidden_state, output_state, probabilities_state = {}, {}, {}, {}
    hidden_state[-1] = np.copy(hidden_previous)
    loss = 0

    # Forward pass
    for t in xrange(len(inputs)):
        input_state[t] = np.zeros((vocabulary_size, 1))
        input_state[t][inputs[t]] = 1
        hidden_state[t] = np.tanh(np.dot(base.input_to_hidden, input_state[t]) + np.dot(base.hidden_to_hidden, hidden_state[t - 1]) + base.hidden_bias)
        output_state[t] = np.dot(base.hidden_to_output, hidden_state[t]) + base.output_bias
        probabilities_state[t] = np.exp(output_state[t]) / np.sum(np.exp(output_state[t]))
        loss += -np.log(probabilities_state[t][targets[t], 0]) # Softmax

    # Backward pass
    gradient = RNNModel()
    gradient_hidden_next = np.zeros_like(hidden_state[0])
    for t in reversed(xrange(len(inputs))):
        gradient_output = np.copy(probabilities_state[t])
        gradient_output[targets[t]] -= 1
        gradient.hidden_to_output += np.dot(gradient_output, hidden_state[t].T)
        gradient.output_bias += gradient_output
        gradient_hidden = np.dot(base.hidden_to_output.T, gradient_output) + gradient_hidden_next
        gradient_hidden_raw = (1 - hidden_state[t] * hidden_state[t]) * gradient_hidden
        gradient.hidden_bias += gradient_hidden_raw
        gradient.input_to_hidden += np.dot(gradient_hidden_raw, input_state[t].T)
        gradient.hidden_to_hidden += np.dot(gradient_hidden_raw, hidden_state[t - 1].T)
        gradient_hidden_next = np.dot(base.hidden_to_hidden.T, gradient_hidden_raw)
    for gradient_param in gradient:
        np.clip(gradient_param, -GRADIENT_LIMIT, GRADIENT_LIMIT, out=gradient_param)
    return loss, gradient, hidden_state[len(inputs) - 1]

def plot_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.title(dataset)
    plt.xlabel("iterations (thousands)")
    plt.ylabel("loss")
    plt.savefig(dir + "/loss.png", bbox_inches="tight")
    plt.close()

def sample_text(iteration, previous_hidden, seed_index, n):
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
    text = "".join(index_to_char[index] for index in indices)

    output_file = open(output_dir, "a")
    output_file.write("Iteration: " + str(iteration) + "\n")
    output_file.write(text + "\n")
    output_file.write("\n")
    output_file.close()

def adagrad_update(memory, gradient):
    for base_param, gradient_param, memory_param in zip(base, gradient, memory):
        memory_param += gradient_param * gradient_param
        base_param += -LEARNING_RATE * gradient_param / np.sqrt(memory_param + ADAGRAD_UPDATE_RATE)

def rnn():

    memory = RNNModel()

    smooth_loss = None
    smooth_losses = []
    resets = 0
    iteration = 0
    data_pointer = 0
    while True:
        if data_pointer + SEQUENCE_LENGTH + 1 >= len(input_data) or iteration == 0:
            resets += 1
            hidden_previous = np.zeros((HIDDEN_LAYER_SIZE, 1))
            data_pointer = 0
            print('{{"metric": "reset", "value": {}}}'.format(resets))
        inputs = [char_to_index[char] for char in input_data[data_pointer:data_pointer + SEQUENCE_LENGTH]]
        targets = [char_to_index[char] for char in input_data[data_pointer + 1:data_pointer + SEQUENCE_LENGTH + 1]]

        if iteration % TEXT_SAMPLING_FREQUENCY == 0:
            sample_text(iteration, hidden_previous, inputs[0], TEXT_SAMPLE_LENGTH)

        loss, gradient, hidden_previous = update(inputs, targets, hidden_previous)

        smooth_loss = loss if smooth_loss is None else smooth_loss * LOSS_SMOOTHING_FACTOR + loss * (1 - LOSS_SMOOTHING_FACTOR)

        if iteration % LOGGING_FREQUENCY == 0:
            print('{{"metric": "iteration", "value": {}}}'.format(iteration))
            print('{{"metric": "smooth_loss", "value": {}}}'.format(smooth_loss))
            print('{{"metric": "loss", "value": {}}}'.format(loss))
            smooth_losses.append(smooth_loss)
            plot_loss(smooth_losses)

        adagrad_update(memory, gradient)

        data_pointer += SEQUENCE_LENGTH
        iteration += 1


if __name__ == "__main__":
    rnn()
