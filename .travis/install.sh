#!/usr/bin/env bash

pip install tqdm
pip install pyyaml
pip install tensorboardX
pip install sacrebleu

case "${TORCH_VER}" in
    0.4.1)
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
    ;;
    0.4.0)
    pip install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
    ;;
esac
