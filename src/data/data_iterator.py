import random
from typing import Iterator
import numpy as np
from itertools import count

from src.utils.common_utils import GlobalNames
from .dataset import Record, zip_records

__all__ = [
    'DataIterator'
]

random.seed(GlobalNames.SEED)

DEFAULT_BUFFER_SIZE_FACTOR = 20


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


def batching(records, batch_size, batching_key):
    batches = []
    batch_buffer = []

    num_samples = 0

    if batching_key == "samples":
        for record in records:
            batch_buffer.append(record)
            num_samples += 1

            if num_samples >= batch_size:
                batches.append(Batch.pack(*batch_buffer))

                num_samples = 0
                batch_buffer = []
    else:
        max_len = 0
        for record in records:
            batch_buffer.append(record)

            num_samples += 1
            max_len = max(max_len, record.index)

            if max_len * num_samples >= batch_size:
                batches.append(Batch.pack(*batch_buffer))

                num_samples = 0
                max_len = 0
                batch_buffer = []

    if len(batch_buffer) > 0:
        batches.append(Batch.pack(*batch_buffer))

    return batches


def fill_buffer(data_iter, stop, key):
    records = []

    n_samples = 0
    key_values = 0

    while True:
        try:
            record = next(data_iter)
        except StopIteration:
            break

        records.append(record)

        n_samples += 1
        key_values += record.index

        if key == "samples":
            if n_samples >= stop:
                break
        else:
            if key_values >= stop:
                break

    return records


def numbering_records_iter(record_iter: Iterator[Record]):
    """Numbering iterator from dataset.
    """
    for ii in count():
        try:
            record = next(record_iter)
        except StopIteration:
            break

        yield zip_records(Record(ii, index=-float('inf')), record)


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
                 batching_func="samples",
                 numbering=False
                 ):

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

        if batching_func not in {"samples", "tokens"}:
            raise ValueError("Unknown batching key {0}".format(batching_func))
        self._batching_key = batching_func

        # buffer size for bucketing
        # buffer size is the max number of batches in a buffer
        # if batching key is 'samples', buffer size is 100 times of batch size,
        # else we suppose that their are 50 tokens in one sample and then estimate
        # the number of samples in one batch as self.batch_size // 50

        if buffer_size is None:
            buffer_size = self.batch_size * DEFAULT_BUFFER_SIZE_FACTOR

        self._buffer_size = buffer_size

        self.use_bucket = use_bucket

        self.numbering = numbering

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
        inc_buffer = fill_buffer(self.data_iter, stop=self._buffer_size, key=self._batching_key)

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

        new_batch_buffer = batching(new_buffer, batch_size=batch_size, batching_key=self._batching_key)
        del new_buffer  # release memory

        self.buffer = new_batch_buffer

    @property
    def is_end(self):
        return self._end

    def reset(self):

        self.buffer = []
        data_iter = self.dataset.data_iter()
        if self.numbering:
            data_iter = numbering_records_iter(data_iter)
        self.data_iter = data_iter

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
