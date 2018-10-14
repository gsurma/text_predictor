import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq


class RNNModel:

    def __init__(self,
                 vocabulary_size,
                 batch_size,
                 sequence_length,
                 hidden_layer_size,
                 cells_size,
                 gradient_clip=5.,
                 training=True):

        cells = []
        [cells.append(rnn.LSTMCell(hidden_layer_size)) for _ in range(cells_size)]
        self.cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, sequence_length])
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            softmax_layer = tf.get_variable("softmax_layer", [hidden_layer_size, vocabulary_size])
            softmax_bias = tf.get_variable("softmax_bias", [vocabulary_size])

        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable("embedding", [vocabulary_size, hidden_layer_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        inputs = tf.split(inputs, sequence_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(previous, _):
            previous = tf.matmul(previous, softmax_layer) + softmax_bias
            previous_symbol = tf.stop_gradient(tf.argmax(previous, 1))
            return tf.nn.embedding_lookup(embedding, previous_symbol)

        with tf.variable_scope("rnn", reuse=tf.AUTO_REUSE):
            outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, loop_function=loop if not training else None)
            output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_layer_size])

        self.logits = tf.matmul(output, softmax_layer) + softmax_bias
        self.probabilities = tf.nn.softmax(self.logits)

        loss = legacy_seq2seq.sequence_loss_by_example([self.logits], [tf.reshape(self.targets, [-1])], [tf.ones([batch_size * sequence_length])])

        with tf.name_scope("cost"):
            self.cost = tf.reduce_sum(loss) / batch_size / sequence_length
        self.final_state = last_state
        self.learning_rate = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_vars), gradient_clip)

        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, trainable_vars))

        tf.summary.histogram("logits", self.logits)
        tf.summary.histogram("probabilitiess", self.probabilities)
        tf.summary.histogram("loss", loss)
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("learning_rate", self.learning_rate)

    def sample(self, sess, chars, vocabulary, length):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        text = ""
        char = chars[0]
        for _ in range(length):
            x = np.zeros((1, 1))
            x[0, 0] = vocabulary[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probabilities, state] = sess.run([self.probabilities, self.final_state], feed)
            probability = probabilities[0]
            total_sum = np.cumsum(probability)
            sum = np.sum(probability)
            sample = int(np.searchsorted(total_sum, np.random.rand(1) * sum))
            predicted = chars[sample]
            text += predicted
            char = predicted
        return text


















