import torch
import torch.nn as nn

def default_init(tensor):
    if tensor.ndimension() == 1:
        nn.init.constant_(tensor, val=0.0)
    else:
        nn.init.xavier_normal_(tensor)

    return tensor

def embedding_init(tensor, val=0.1):
    nn.init.uniform_(tensor, -val, val)

    return tensor

def rnn_init(tensor):
    if tensor.ndimension() != 2:
        return default_init(tensor)

    return default_init(tensor)

    r, c = tensor.size()

    if r % c == 0:
        dim = 0
        n = r // c
        sub_size = (c, c)
    elif c % r == 0:
        dim = 1
        n = c // r
        sub_size = (r, r)
    else:
        return default_init(tensor)

    sub_tensors = [torch.Tensor(*sub_size).normal_(0, 1) for _ in range(n)]
    sub_tensors = [torch.svd(w, some=True)[0] for w in sub_tensors]

    init_tensor = torch.cat(sub_tensors, dim=dim) # [r, c]

    with torch.no_grad():  # inplace op should be wrapped in no_grad mode.
        tensor.copy_(init_tensor)

    return tensor


