import codecs
import os
import collections
import numpy as np


class DataLoader:
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = seq_length

        self.preprocess(os.path.join(data_dir, "input.txt"))
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file):
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocab.get, data)))

    def create_batches(self):
        self.batches_size = int(self.tensor.size / (self.batch_size *
                                                    self.sequence_length))
        print "Tensor size: " + str(self.tensor.size)
        print "Batch size: " + str(self.batch_size)
        print "Sequence length: " + str(self.sequence_length)
        print "Batches size: " + str(self.batches_size)
        print ""

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.batches_size == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        # reshape the original data into the length self.num_batches * self.batch_size * self.seq_length for convenience.
        self.tensor = self.tensor[:self.batches_size * self.batch_size * self.sequence_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)

        #ydata is the xdata with one position shift.
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.batches_size, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.batches_size, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
