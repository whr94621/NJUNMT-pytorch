# README

## Data Preprocess

1. Build dictionary

See help of ```./data/build_dictionary.py```

2. Configure samples

See ```./configs/test_config.yaml```

## Requirements

- Pytorch (0.3.1)

- tqdm (for progressbar)

- tensorboardX (for tensorboard)

## Q&A

1. What is ```shard_size``` ?

```shard_size``` is trick borrowed from OpenNMT-py, which

could make large model run in the memory-limited condition.

For example, you can run wmt17 EN2DE task on a 8GB GTX1080 card

with batch size 64 by setting ```shard_size=10```

2. What is ```use_bucket``` ?

When using bucket, parallel sentences will be sorted partially
according to the length of target sentence.

Set this option to ```true``` will bring considerable improvement

on training speed, but I find this will bring slower and unstable descent of

loss curve. (Reason is still unknown)



