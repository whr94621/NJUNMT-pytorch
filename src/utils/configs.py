# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def add_default_configs(configs: dict, default_configs: dict):
    """
    Add default items to current configuration
    """
    for key, value in default_configs.items():
        if key not in configs:
            configs[key] = value
        elif isinstance(default_configs[key], dict) and isinstance(configs[key], dict):
            add_default_configs(configs[key], default_configs[key])
        else:
            continue

    return configs


def default_base_configs():
    return {
        "data_configs": {},
        "model_configs": {
            "label_smoothing": 0.0,  # close label smoothing default.

        },
        "optimizer_configs": {
            "grad_clip": -1.0,  # Close gradient clipping given a non-positive value.
            "optimizer_params": None,
            "schedule_method": None,
            "scheduler_configs": None,
        },
        "training_configs": {
            "seed": 1234,
            "max_epochs": 10000,
            "shuffle": False,
            "use_bucket": True,
            "batching_key": "samples",
            "buffer_size": 5000,
            "update_cycle": 1,
            "norm_by_words": False,
            "num_kept_checkpoints": 1,
            "num_kept_best_model": 1,
            "bleu_valid_configs": {
                "max_steps": 150,
                "beam_size": 5,
                "alpha": 0.0,
                "postprocess": False,
                "sacrebleu_args": ""
            },
            "bleu_valid_warmup": 0,
            "early_stop_patience": 100000,
            "bleu_valid_batch_size": 5,

            # About moving average
            "moving_average_method": None,  # sma | ema | None
            "moving_average_alpha": 0.0
        }
    }


def add_user_configs(default_configs: dict, user_configs: dict) -> dict:
    """Add user defined configurations to default configurations"""
    add_default_configs(user_configs, default_configs)
    return user_configs


def pretty_configs(user_configs: dict, prefix=""):
    """Format configurations"""
    output = []

    for key, value in user_configs.items():
        if not isinstance(value, dict):
            output.append("{0}{1}: {2}".format(prefix, key, value))
        else:
            output.append("{0}{1}:\n{2}".format(prefix, key, pretty_configs(value, prefix + "  ")))

    return '\n'.join(output)
