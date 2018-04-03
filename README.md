# README

This is our in-house implementation of Transformer model in "[Attenion is all you need](https://arxiv.org/abs/1706.03762)".

# Requirements

- python 3.5+
- pytorch 3.1
- tqdm
- tensorboardX

# Usage

## Data Preprocessing

See help of ```./data/build_dictionary.py```

Vocabulary will be stored as json format.

We highly recommend not to set the limitation of the number of
words and control it by config files while training.

## Configuration

See examples in ```./configs``` folder. You can reproduce our
Chinese-to-English Baseline by directly using those configures.

```loss_schedule_config.yaml``` is the configure file using
schedule moethod in.

```noam_schedule_config.yaml``` is the configure file using
valid loss as schedule criterion.

## Training
See training script ```./scripts/train.sh```

## Translation
See translation script ```./scripts/translation.sh```

# Performance

| Decay Method | Use Bucket | MT03(dev) | MT04  | MT05  | MT06  |
|--------------|------------|-----------|-------|-------|-------|
| Loss         | TRUE       | 40.22     | 41.61 | 37.17 | 35.39 |
| Loss         | FALSE      | 41.48     | 42.31 | 39.43 | 36.85 |
| Noam         | TRUE       | 40.50     | 41.90 | 38.19 | 36.12 |
| Noam         | FALSE      | 41.80     | 42.52 | 39.05 | 36.90 |

# Q&A

1. What is ```shard_size``` ?

```shard_size``` is trick borrowed from OpenNMT-py, which

could make large model run in the memory-limited condition.

For example, you can run wmt17 EN2DE task on a 8GB GTX1080 card

with batch size 64 by setting ```shard_size=10```

2. What is ```use_bucket``` ?

When using bucket, parallel sentences will be sorted partially
according to the length of target sentence.

Set this option to ```true``` will bring considerable improvement
but performance regression.

# Acknowledgement

- This code is heavily borrowed from OpenNMT/OpenNMT-py and have been
simplified for research use.




