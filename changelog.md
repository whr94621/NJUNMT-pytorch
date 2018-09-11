# Change Log

## WIP

### New Features - Delayed Update

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

