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
LEARNING_RATE = 0.001
ADAGRAD_UPDATE_RATE = 1e-8
LOGGING_FREQUENCY = 1000
TEXT_SAMPLING_FREQUENCY = 1000
GRADIENT_LIMIT = 5
RANDOM_WEIGHT_INIT_FACTOR = 1e-2
LOSS_SMOOTHING_FACTOR = 0.999
TEXT_SAMPLE_LENGTH = 200


# class RNNSubLayer:
#
#     def __init__(self, random_init=False):
#         self.input_to_hidden = (np.random.randn(HIDDEN_LAYER_SIZE, vocabulary_size) * RANDOM_WEIGHT_INIT_FACTOR) if random_init else np.zeros((HIDDEN_LAYER_SIZE, vocabulary_size))
#         self.hidden_to_hidden = (np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * RANDOM_WEIGHT_INIT_FACTOR) if random_init else np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
#         self.hidden_to_output = (np.random.randn(vocabulary_size, HIDDEN_LAYER_SIZE) * RANDOM_WEIGHT_INIT_FACTOR) if random_init else np.zeros((vocabulary_size, HIDDEN_LAYER_SIZE))
#         self.hidden_bias = np.zeros((HIDDEN_LAYER_SIZE, 1))
#         self.output_bias = np.zeros((vocabulary_size, 1))
#
#     def __iter__(self):
#         return iter([self.input_to_hidden, self.hidden_to_hidden, self.hidden_to_output, self.hidden_bias, self.output_bias])


class RNNLayer:

    def __init__(self, x_size, h_size, y_size):
        self.h_size = h_size

        self.x = {}
        self.h = {}
        self.h_last = np.zeros((h_size, 1))

        # self.base = RNNSubLayer(random_init=True)
        # self.memory = RNNSubLayer()
        # self.gradient = RNNSubLayer()

        self.W_xh = np.random.randn(h_size, x_size)*RANDOM_WEIGHT_INIT_FACTOR
        self.W_hh = np.random.randn(h_size, h_size)*RANDOM_WEIGHT_INIT_FACTOR
        self.W_hy = np.random.randn(y_size, h_size)*RANDOM_WEIGHT_INIT_FACTOR
        self.b_h = np.zeros((h_size, 1))
        self.b_y = np.zeros((y_size, 1))

        self.adaW_xh = np.zeros((h_size, x_size))
        self.adaW_hh = np.zeros((h_size, h_size))
        self.adaW_hy = np.zeros((y_size, h_size))
        self.adab_h = np.zeros((h_size, 1))
        self.adab_y = np.zeros((y_size, 1))

    #given an input, step the internal state and return the output of the network
    #Because the whole network is together in one object, I can make it easy and just
    #take a list of input ints, transform them to 1-of-k once, and prop everywhere.
    #
    #   Here is a diagram of what's happening. Useful to understand backprop too.
    #
    #                  [b_h]                                              [b_y]
    #                    v                                                  v
    #   x -> [W_xh] -> [sum] -> h_raw -> [nonlinearity] -> h -> [W_hy] -> [sum] -> y ... -> [e] -> p
    #                    ^                                 |
    #                    '----h_next------[W_hh]-----------'
    #
    def forward_pass(self, x):
        #load the last state from the last batch in to the beginning of h
        #it is necessary to save it outside of h because h is used in backprop
        self.h[-1] = self.h_last
        self.x = x

        y = {}
        p = {}#p[t] = the probabilities of next chars given chars passed in at times <=t
        for t in range(len(self.x)):#for each moment in time

            #find new hidden state in this layer at this time
            self.h[t] = np.tanh(np.dot(self.W_xh, self.x[t]) + \
                np.dot(self.W_hh, self.h[t-1]) + self.b_h)#tanh

            #find unnormalized log probabilities for next chars
            y[t] = np.dot(self.W_hy, self.h[t]) + self.b_y#output from this layer is input to the next
            p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))#find probabilities for next chars

        #save the last state from this batch for next batch
        self.h_last = self.h[len(x)-1]

        return y, p

    #given the RNN a sequence of correct outputs (seq_length long), use
    #them and the internal state to adjust weights
    def backward_pass(self, dy):

        #we will need some place to store gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.h_size, 1))#I think this is the right dimension
        dx = {}

        for t in reversed(range(len(dy))):
        #     dy = np.copy(ps[t])
        #     dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        #     dWhy += np.dot(dy, hs[t].T)
        #     dby += dy
        #     dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        #     dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        #     dbh += dhraw
        #     dWxh += np.dot(dhraw, xs[t].T)
        #     dWhh += np.dot(dhraw, hs[t - 1].T)
        #     dhnext = np.dot(Whh.T, dhraw)
        # for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        #     np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
            #find updates for y stuff
            dW_hy += np.dot(dy[t], self.h[t].T)
            db_y += dy[t]

            #backprop into h and through nonlinearity
            dh = np.dot(self.W_hy.T, dy[t]) + dh_next
            dh_raw = (1 - self.h[t]**2)*dh#tanh

            #find updates for h stuff
            dW_xh += np.dot(dh_raw, self.x[t].T)
            dW_hh += np.dot(dh_raw, self.h[t-1].T)
            db_h += dh_raw

            #save dh_next for subsequent iteration
            dh_next = np.dot(self.W_hh.T, dh_raw)

            #save the error to propagate to the next layer. Am I doing this correctly?
            dx[t] = np.dot(self.W_xh.T, dh_raw)

        #clip to mitigate exploding gradients
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -GRADIENT_LIMIT, GRADIENT_LIMIT, out=dparam)

        for param, dparam, adaparam in zip([self.W_hh, self.W_xh, self.W_hy, self.b_h, self.b_y], \
                                            [dW_hh, dW_xh, dW_hy, db_h, db_y], \
                                            [self.adaW_hh, self.adaW_xh, self.adaW_hy, self.adab_h, self.adab_y]):
            adaparam += dparam*dparam
            param += -LEARNING_RATE*dparam/np.sqrt(adaparam+ADAGRAD_UPDATE_RATE)

        return dx

def rnn():
    rnn1 = RNNLayer(vocabulary_size, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
    rnn2 = RNNLayer(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
    rnn3 = RNNLayer(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE, vocabulary_size)

    smooth_loss = None
    smooth_losses = []

    resets = 0
    iteration = 0
    data_pointer = 0

    while True:
        if data_pointer + SEQUENCE_LENGTH + 1 >= len(input_data) or iteration == 0:
            resets += 1
            data_pointer = 0
            rnn1.h_last = np.zeros((rnn1.h_size, 1))
            rnn2.h_last = np.zeros((rnn2.h_size, 1))
            rnn3.h_last = np.zeros((rnn3.h_size, 1))
            print('{{"metric": "reset", "value": {}}}'.format(resets))
        inputs = [char_to_index[char] for char in input_data[data_pointer:data_pointer + SEQUENCE_LENGTH]]
        targets = [char_to_index[char] for char in input_data[data_pointer + 1:data_pointer + SEQUENCE_LENGTH + 1]]

        if iteration % TEXT_SAMPLING_FREQUENCY == 0:
            sample_text([rnn1, rnn2, rnn3], inputs[0], iteration)

        # Forward pass
        x = one_of_k(inputs, len(chars))
        y1, p1 = rnn1.forward_pass(x)
        y2, p2 = rnn2.forward_pass(y1)
        y3, p3 = rnn3.forward_pass(y2)

        loss = 0
        for t in range(len(targets)):
            loss += -np.log(p3[t][targets[t], 0])

        smooth_loss = loss if smooth_loss is None else smooth_loss * LOSS_SMOOTHING_FACTOR + loss * (1-LOSS_SMOOTHING_FACTOR)

        if iteration % LOGGING_FREQUENCY == 0:
            smooth_losses.append(smooth_loss)
            plot(smooth_losses, "loss")
            print('{{"metric": "iteration", "value": {}}}'.format(iteration))
            print('{{"metric": "smooth_loss", "value": {}}}'.format(smooth_loss))

        # Backward pass
        dy = logprobs(p3, targets)
        dx3 = rnn3.backward_pass(dy)
        dx2 = rnn2.backward_pass(dx3)
        dx1 = rnn1.backward_pass(dx2)

        data_pointer += SEQUENCE_LENGTH
        iteration += 1


def plot(data, y_label):
    plt.plot(range(len(data)), data)
    plt.title(dataset)
    plt.xlabel("iterations (thousands)")
    plt.ylabel(y_label)
    plt.savefig(dir + "/" + y_label + ".png", bbox_inches="tight")
    plt.close()

def sample_text(rnns, seed, iteration):
    indices = []
    index = seed
    for t in range(TEXT_SAMPLE_LENGTH):
        x = one_of_k([index], vocabulary_size)
        for i in range(len(rnns)):
            x, p = rnns[i].forward_pass(x)
        index = np.random.choice(range(len(p[0])), p=p[0].ravel())
        indices.append(index)
    text = ''.join([index_to_char[n] for n in indices])
    output_file = open(output_dir, "a")
    output_file.write("Iteration: " + str(iteration) + "\n")
    output_file.write(text + "\n")
    output_file.write("\n")
    output_file.close()

def logprobs(p, targets):
    dy = {}
    for t in range(len(targets)):
        dy[t] = np.copy(p[t])
        dy[t][targets[t]] -= 1
    return dy

def one_of_k(inputs, k):
    x = {}
    for t in range(len(inputs)):
        x[t] = np.zeros((k, 1))
        x[t][inputs[t]] = 1
    return x


if __name__ == "__main__":
    rnn()
