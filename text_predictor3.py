from __future__ import print_function
import argparse
import tensorflow as tf
from utils import TextLoader
from model import Model

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument('--data_dir', type=str, default='data/sherlock',
                    help='data directory containing input.txt with training examples')

SAVE_DIR = "save"
SAMPLE_LENGTH = 500

parser.add_argument('--model', type=str, default='lstm',
                    help='lstm, rnn, gru, or nas')
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
# Optimization
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN sequence length. Number of timesteps to unroll for.')
parser.add_argument('--batch_size', type=int, default=50,
                    help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='decay rate for rmsprop')

args = parser.parse_args()


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    model = Model(args)

    with tf.Session() as sess:

        summaries = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())

        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                _, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed)
                iteration = e * data_loader.num_batches + b
                print("{}/{} (epoch {}), train_loss = {:.3f}"
                      .format(iteration,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss))
                if iteration % 1000 == 0:
                    sample(sess, args, data_loader.chars, data_loader.vocab)

def sample(sess, args, chars, vocab):
    model = Model(args, training=False)
    print(model.sample(sess, chars, vocab, SAMPLE_LENGTH, chars[0]).encode('utf-8'))


if __name__ == '__main__':
    train(args)
























