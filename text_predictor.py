import tensorflow as tf
from data_provider import DataProvider
from rnn_model import RNNModel
import sys
import matplotlib
import numpy as np
import time
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Args
if len(sys.argv) != 2:
    print "Please select a dataset."
    print "Usage: python text_predictor.py <dataset>"
    print "Available datasets: shakespeare, wikipedia, reuters, hackernews, wikipedia, war_and_peace, sherlock"
    exit(1)
else:
    dataset = sys.argv[1]
print "Selected dataset: " + str(dataset)

# I/O
data_dir = "./data/" + dataset
tensorboard_dir = data_dir + "/tensorboard/" + str(time.strftime("%Y-%m-%d_%H-%M-%S"))
input_file = data_dir + "/input.txt"
output_file = data_dir + "/output.txt"
output = open(output_file, "w")
output.close()

# Hyperparams
BATCH_SIZE = 32
SEQUENCE_LENGTH = 50
LEARNING_RATE = 0.01
DECAY_RATE = 0.97
HIDDEN_LAYER_SIZE = 256
CELLS_SIZE = 2

TEXT_SAMPLE_LENGTH = 200
SAMPLING_FREQUENCY = 500
LOGGING_FREQUENCY = 1000


def rnn():
    data_loader = DataProvider(data_dir, BATCH_SIZE, SEQUENCE_LENGTH)
    model = RNNModel(data_loader.vocabulary_size, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, hidden_layer_size=HIDDEN_LAYER_SIZE, cells_size=CELLS_SIZE)

    with tf.Session() as sess:

        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        writer.add_graph(sess.graph)
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
                for index, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[index].c
                    feed[h] = state[index].h

                iteration = epoch * data_loader.batches_size + batch
                summary, loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                writer.add_summary(summary, iteration)
                temp_losses.append(loss)

                if iteration % SAMPLING_FREQUENCY == 0:
                    sample_text(sess, data_loader, iteration)

                if iteration % LOGGING_FREQUENCY == 0:
                    print("Iteration: {}, epoch: {}, loss: {:.3f}".format(iteration, epoch, loss))
                    smooth_losses.append(np.mean(temp_losses))
                    temp_losses = []
                    plot(smooth_losses, "loss")
            epoch += 1

def sample_text(sess, data_loader, iteration):
    model = RNNModel(data_loader.vocabulary_size, batch_size=1, sequence_length=1, hidden_layer_size=HIDDEN_LAYER_SIZE, cells_size=CELLS_SIZE, training=False)
    text = model.sample(sess, data_loader.chars, data_loader.vocabulary, TEXT_SAMPLE_LENGTH, data_loader.chars[0]).encode('utf-8')
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
























