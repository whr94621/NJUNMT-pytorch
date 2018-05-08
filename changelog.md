# adaptation for pytorch 0.4.0

## Done
1. replace `nn.Relu(inplace=False)` with `nn.Relu()`, for inplace op modifies input tensor, leading to `"One of the variables needed for gradient computation has been modified by an inplace operation"` Exception.
2. remove `isinstance(a_tensor, Variable)`, which is currently always `True`.
3. user-defined init functions should receive `tensor` as input instead of `tensor.data`
4. user-defined init functions with inplace operation (e.g., `tensor.copy_()`) should be wrapped by context mannager `with torch.no_grad(True)`.
4. built-in init functions turn to inplace operation, whose api have been changed from `some_init_fn()` to `some_init_fn_()`
4. use `torch.tensor(requires_grad=bool)` instead of `Variable`
5. use `torch.set_grad_enabled(is_train)` instead of `volatile` Flag for `Variable`
6. use `tensor.detach()` instead of `tensor.data`
7. use `scalar_tensor.item()` instead of `scalar_tensor.data[0]`
8. `ONNX`-related functions (e.g., `torch.nn.utils.rnn .pack_padded_sequence()`) should not be called by explicit keyword args like `pack_padded_sequence(input=input_sorted, lengths=slens, batch_first=self.batch_first)`, which leads to failed arg check in `might_trace()` in `torch.onnx._symbolic_override_wrapper_maker`.

