# Benchmark of NJUNMT-pytorch

## DL4MT

### Chinese to English

See configuration in  ```dl4mt_config.yaml```

| **Decay Method** | **Granularity** | **MT03(Dev)** | **MT04**  | **MT05**  | **MT06**  |
|--------------|-------------|-----------|-------|-------|-------|
| Loss         | Word(30K)   | 38.84     | 41.02 | 36.46 | 35.26 |
| Loss         | BPE(30K)    | 37.72     | 38.64 | 35.09 | 33.73 |
| Noam         | Word(30K)   | 38.31     | 39.82 | 35.84 | 33.96 |
| Noam         | BPE(30K)    | 38.48     | 40.47 | 36.79 | 35.21 |

Word(30K): Training NMT at word level and keep most 30K frequent words and keep the rest as a special token <UNK>.

BPE(30K): Use [Byte Pair Encoding](https://github.com/rsennrich/subword-nmt) to split words into subword sequences.
We do 30K BPE operations here and keep all the BPE tokens.

When choosing ```Loss``` as learning rate method, BPE model performs abnormally worse than word-level model. This result
is confusing and one of the possible reasons maybe that the first occurrence of decay is too late, which make this scheduling
policy degenerate into the vanilla Adam.


## Transformer

## Chinese to English

Learning rate decay method as well as bucket will
bring different results.

| Decay Method | Use Bucket | MT03(dev) | MT04  | MT05  | MT06  |
|--------------|------------|-----------|-------|-------|-------|
| Loss         | TRUE       | 40.22     | 41.61 | 37.17 | 35.39 |
| Loss         | FALSE      | 41.48     | 42.31 | 39.43 | 36.85 |
| Noam         | TRUE       | 40.50     | 41.90 | 38.19 | 36.12 |
| Noam         | FALSE      | 41.80     | 42.52 | 39.05 | 36.90 |