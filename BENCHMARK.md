# Benchmark of NJUNMT-pytorch

## DL4MT

### Chinese to English

See configuration in  ```dl4mt_config.yaml```

| MT03(dev) | MT04  | MT05  | MT06  |
|-----------|-------|-------|-------|
| 38.84     | 40.85 | 36.61 | 37.59 |

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