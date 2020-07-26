# Conformer
Pytorch implementation of Conformer Block.
[paper](https://arxiv.org/abs/2005.08100)

## Usage
You can use this Conformer blockr with the following codes.  
```
import torch
import json
from CF import get_conformer

conf = json.load(open('conformer.conf'))
net = get_conformer(**conf)
net.eval()

data = torch.randn(1, 32, conf['d_model'])
# data should be formatted as (B, L, D)
# B as batch-size, L as sequence-length, D as feature-dimension.

out = net(data)
```
The shape of output is (B, L, D)
