# Change Log

## 2018/10/09

### Improvments

- Save k-best models.
- Add ```dim_per_head``` option for transformer. Now ```dim_per_head``` * ```n_head``` can not equal to ```d_model```.
- Add exponential moving average (EMA).
- Support pytorch 0.4.1

## 2018/9/30

### New Features - Different way to batch
This feature enables you to use different way to batch your data. We provide two method, "samples" and "tokens". "samples" means how many bi-text pairs (samples) in one batch, while "tokens" means how many tokens in one batch (if there are several sentences in one sample, this means most tokens among them). You can use these two kinds of method by setting "batching_key" as "samples" or "tokens".

### New Features - Delayed Update
This feature enables you to emuluate multi GPUs on a single GPU. By setting ```update_cycle``` as a value larger than 1, the model will compute forward and accumulate gradients for this many steps before parameters update, which behaves like one update step with actual batch size as ```update_cycle * batch_size```. For example, if we want to use 25000 tokens in a batch on a single 1080 GPU(8GB Mem), we can set ```batch_size``` as 1250 and ```update_cycle``` as 20. This will prevent OOM problem.

### Improvments

- Put source sentences with similar length into a batch during inference. This significantly improve the speed of beam search(beam size is 5) from 497.01 tokens/sec to 1179.05 tokens/sec on a single 1080ti GPU.
- Use SacreBLEU to compute BLEU scores instead of ```multi-bleu.perl``` and ```multi-bleu-detok.perl```.
- Add ```AdamW``` and ```Adafactor```
- Count the number of pads when using "tokens" as batching key.
- Add ensemble decoding with different checkpoints.
- Add length penalty when decoding.

### Bugs fix
- Mask padding in generator.
- RAM will not continue increasing during training.
## 2018/07/16

### New Features - Travis CI Enable
We add components for travis ci. Now all tests can only run on CPU.

### New Features - Standalone Logging Module
We combine functions about logging into a standalone module. Now we can redirect logging info to files when using ```tqdm``` at the same time.

### New Features - Use External Scritps to Evaluate BLEU
We integrate scripts from [Moses](https://github.com/moses-smt/mosesdecoder), including tokenizing, recasing and bleu calculation.

### API Changes

- Refactor RNN for data parallelism support.

### Bugs fix

- close file handles when shuffling is over
- fix the typo of ```Criterion```

## 2018/05/08

### Upgrade to pytorch 0.4.0, drop 0.3.1 support

- replace `nn.Relu(inplace=False)` with `nn.Relu()`, for inplace op modifies input tensor, leading to `"One of the variables needed for gradient computation has been modified by an inplace operation"` Exception.
- remove `isinstance(a_tensor, Variable)`, which is currently always `True`.
- user-defined init functions should receive `tensor` as input instead of `tensor.data`
- user-defined init functions with inplace operation (e.g., `tensor.copy_()`) should be wrapped by context mannager `with torch.no_grad(True)`.
- built-in init functions turn to inplace operation, whose api have been changed from `some_init_fn()` to `some_init_fn_()`
- use `torch.tensor(requires_grad=bool)` instead of `Variable`; use `torch.set_grad_enabled(is_train)` instead of `volatile` Flag for `Variable`;
use `tensor.detach()` instead of `tensor.data`; use `scalar_tensor.item()` instead of `scalar_tensor.data[0]`
- `ONNX`-related functions (e.g., `torch.nn.utils.rnn .pack_padded_sequence()`) should not be called by explicit keyword args like `pack_padded_sequence(input=input_sorted, lengths=slens, batch_first=self.batch_first)`, which leads to failed arg check in `might_trace()` in `torch.onnx._symbolic_override_wrapper_maker`.

