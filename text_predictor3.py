from __future__ import print_function
import tensorflow as tf
from data_loader import DataLoader
from model import Model
import sys
import matplotlib
import numpy as np
from statistics import mean
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Args
if len(sys.argv) != 2:
    print("Please select a dataset.")
    print("Usage: python text_predictor.py <dataset>")
    print("Available datasets: shakespeare, wikipedia, reuters, hackernews, wikipedia, war_and_peace")
    exit(1)
else:
    dataset = sys.argv[1]
print("Selected dataset: " + str(dataset))


# I/O
data_dir = "./data/" + dataset
input_file = data_dir + "/input.txt"

output_file = data_dir + "/output.txt"
output = open(output_file, "w")
output.close()

# Hyperparams
SAMPLE_LENGTH = 500
SAMPLING_FREQUENCY = 1000
LOGGING_FREQUENCY = 1000
BATCH_SIZE = 50
SEQUENCE_LENGTH = 50
LEARNING_RATE = 0.02
DECAY_RATE = 0.97

def rnn():
    data_loader = DataLoader(data_dir, BATCH_SIZE, SEQUENCE_LENGTH)
    model = Model(data_loader.vocab_size, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH)

    with tf.Session() as sess:

        summaries = tf.summary.merge_all() #TODO: needed?
        sess.run(tf.global_variables_initializer())

        epoch = 0
        temp_losses = []
        smooth_losses = []

        while True:
            sess.run(tf.assign(model.learning_rate, LEARNING_RATE * (DECAY_RATE ** epoch)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for batch in range(data_loader.batches_size):
                input, output = data_loader.next_batch()
                feed = {model.input_data: input, model.targets: output}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                _, loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                temp_losses.append(loss)

                iteration = epoch * data_loader.batches_size + batch

                if iteration % SAMPLING_FREQUENCY == 0:
                    sample_text(sess, data_loader, iteration)

                if iteration % LOGGING_FREQUENCY == 0:
                    print("Iteration: {}, epoch: {}, loss: {:.3f}".format(iteration, epoch, loss))
                    m = np.mean(temp_losses)
                    smooth_losses.append(m)
                    temp_losses = []
                    plot(smooth_losses, "loss")
            epoch += 1

def sample_text(sess, data_loader, iteration):
    model = Model(data_loader.vocab_size, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, training=False)
    text = model.sample(sess, data_loader.chars, data_loader.vocab, SAMPLE_LENGTH, data_loader.chars[0]).encode('utf-8')
    output = open(output_file, "a")
    output.write("Iteration: " + str(iteration) + "\n")
    output.write(text + "\n")
    output.write("\n")
    output.close()

def plot(data, y_label):
    plt.plot(range(len(data)), data)
    plt.title(dataset)
    plt.xlabel("iterations (thousands)")
    plt.ylabel(y_label)
    plt.savefig(data_dir + "/" + y_label + ".png", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    rnn()
























