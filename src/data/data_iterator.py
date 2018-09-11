from .dataset import Record
import numpy as np
import random

from src.utils.common_utils import GlobalNames

__all__ = [
    'DataIterator'
]

random.seed(GlobalNames.SEED)


class Batch(object):
    """
    'Batch' is a list of 'Record's which can coalesce into one batch.

    'content' is a list of records which will be packed into one batch
    """

    def __init__(self, *records):

        self.content = list(records)

    def unpack(self):
        """ Unpack a 'Batch' instance into batched data.

        records in a batch will be split into several list according to the number of
        fields. For example, if a batch has three records R1, R2, R3. Ri has two fields, the
        the value of which are [a, b], then the result of unpack will be two lists, i.e.
        [a1, a2, a3], [b1, b2, b3]
        """
        n_fields = self.content[0].n_fields  # all the records must have the same field

        outs = tuple([r.fields[ii] for r in self.content] for ii in range(n_fields))

        if n_fields == 1:
            return outs[0]
        else:
            return outs

    @classmethod
    def pack(cls, *records: Record) -> 'Batch':
        """
        Pack a list of records into a batch.
        """

        return cls(*records)


def batchify(buffer, batch_size, batching_func):
    """
    Batchify buffer, a list of records, given a ```batching_func``` and ```batch_size```.
    """
    batches = []

    for records in accumulate_takewhile(buffer, batch_size, batching_func):
        batches.append(Batch.pack(*records))

    return batches


class accumulate_takewhile(object):
    """
    This is the combination of ```itertools.takewhile``` and ```itertools.accumulate```
    >>> my_iter = accumulate_takewhile(range(10), 3)
    >>> list(my_iter) # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    def __init__(self, iterable, stop, func=lambda item: 1):

        self.iter = iter(iterable)
        self.func = func
        self.size = stop

    def __iter__(self):
        return self

    def __next__(self):

        out = []
        count = 0

        while True:
            try:
                item = next(self.iter)
            except StopIteration:
                if len(out) > 0:
                    return out
                else:
                    raise StopIteration

            out.append(item)
            count += self.func(item)

            if count >= self.size:
                return out


def accumulate_slicewhilce(data_iter, stop, key_func=lambda _: 1):
    """Slicing data according to key function

    Accumulate data into one batch until the accumulated value of key function
    reach stop criterion.
    """

    lines = []
    count = 0
    while True:
        try:
            line = next(data_iter)
        except StopIteration:
            break

        lines.append(line)
        count += key_func(line)

        if count >= stop:
            break

    return lines


def add_noise_to_length(lengths, noise=1.0):
    """Add noise to the length of sequences.

    Args:
        lengths: The length of sequences.
        noise_ratio: The ratio to add noise to the lengths.
    """

    noisy_lengths = [l + np.random.uniform(- noise, noise) for l in lengths]

    return noisy_lengths


class DataIterator(object):
    """
    ```DataIterator``` defines the way to group your data into a batch. You can choose the way to batchify your data.
    In current implementation, we only provide "samples" and "tokens", which are the two main methods in machine
    translation.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 buffer_size=None,
                 use_bucket=True,
                 batching_func="samples"):

        """ Build data iterator given a dataset

        Args:
            dataset: An Dataset Object
            batch_size: Integer. Size of a batch. When batching_key is "samples", it represents the
                the number of samples. When batching_key is "tokens", it represents the tokens in a batch.
            use_bucket: Boolean value. Whether to use bucket.
            batching_key: Criterion to allocate a batch. Can only be "samples" or "tokens"
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Batching Key
        #
        # We have two kinds of batching key, ```tokens``` and ```samples```.
        # For tokens, we allocate a batch according to the number of tokens in it. For example,
        # in machine translation, if we use "tokens" as the key and set the batch_size as 4096,
        # we allocate a batch when the number of tokens at source or target side reach 4096.
        # For samples, we allocate a batch according to the number of samples in it. In machine
        # translation, 50 batch size with "samples" as key means 50 bi-text sentences.

        if batching_func == "samples":
            self.batching_func = lambda line: 1
        elif batching_func == "tokens":
            self.batching_func = lambda record: record.index
        else:
            assert callable(batching_func)
            self.batching_func = batching_func

        # buffer size for bucketing
        # buffer size is the max number of batches in a buffer
        # if batching key is 'samples', buffer size is 100 times of batch size,
        # else we suppose that their are 50 tokens in one sample and then estimate
        # the number of samples in one batch as self.batch_size // 50

        if buffer_size is None:
            buffer_size = self.batch_size * 10

        self._buffer_size = buffer_size

        self.use_bucket = use_bucket

        self.reset()

    def __len__(self):
        return len(self.dataset)

    @property
    def n_datasets(self):
        return self.dataset.n_fields

    def _fill_buffer(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        # 1. Allocate a new buffer
        inc_buffer = accumulate_slicewhilce(self.data_iter, self._buffer_size, key_func=self.batching_func)

        if len(inc_buffer) <= 0:
            # data_iter reach the end of the dataset
            self._end = True
            return

        # 2. Merge the residual samples in previous buffer (if any) into the inc_buffer
        if len(self.buffer) > 0:
            new_buffer = self.buffer[0].content + inc_buffer
        else:
            new_buffer = inc_buffer

        # 3. Split buffer into batches. If ues_bucket is enable,
        # we sort the whole buffer according to the length of the sentence.
        # In order to randomize the process of batching, we add a little bit noise on the length.

        if self.use_bucket:
            scores = np.array([record.index for record in new_buffer])
            noisy_scores = add_noise_to_length(scores)
            sorted_indices = np.argsort(noisy_scores).tolist()
            new_buffer = [new_buffer[i] for i in sorted_indices]

        new_batch_buffer = batchify(new_buffer, batch_size=batch_size, batching_func=self.batching_func)
        del new_buffer  # release memory

        self.buffer = new_batch_buffer

    @property
    def is_end(self):
        return self._end

    def reset(self):
        self.buffer = []
        self.data_iter = self.dataset.data_iter()
        self._end = False

    def build_generator(self, batch_size=None):

        while True:

            # We re-allocate the buffer when there at most on batch.
            # Usually this batch is not full.

            if len(self.buffer) <= 1:
                self._fill_buffer(batch_size=batch_size)

            if len(self.buffer) == 0:
                """Reach the end of the dataset, exit.
                """
                self.reset()
                break

            # Accumulated batches until reach the batch_size

            try:
                batch = self.buffer.pop(0)
            except IndexError:
                self.reset()
                break

            yield batch.unpack()
