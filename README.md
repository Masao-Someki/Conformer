# Conformer implementation with Pytorch
Pytorch implementation of Conformer.

You can use this block to build your own great model!!

## Whats new
- 2021/06/13 Supported KMeans Attention for multi-head module.

## Model Architecture

- Total flow of the Conformer Block

  ![](./for_readme/total.png)

  

- Feed Forward Module

  ![](./for_readme/FeedForwardModule.png)

  

- Multi-Head Self Attention Module

  ![](./for_readme/MHAModule.png)

  

- Convolution Module

  ![](./for_readme/ConvModule.png)



## Requirements

This repository is tested on Ubuntu 20.04 LTS with the following environment.

- Python3.7+
- Cuda10.2
- CuDNN7+



## Setup

You can setup this repository with the following commands

```
cd tools
make
```

Please check if the `venv` directory is successfully located under the tools directory.



## Usage

You can use a Conformer block with the following codes.  

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

The shape of output is (B, L, D).



Or you can use this block in the following way.

```
import torch
from CF import Conformer

net = Conformer(
	d_model=256,
	ff1_hsize=1024,
    	ff1_dropout=0.2,
    	n_head=4,
    	mha_dropout=0.2,
   	kernel_size=3,
    	conv_dropout=0.2,
    	ff2_hsize=1024,
    	ff2_dropout=0.2
)
net.eval()

data = torch.randn(1, 32, 256)
out = net(data)
```

You can use KMeans Attention to reduce memory use.
```
import torch
from CF import Conformer

net = Conformer(
	d_model=256,
	ff1_hsize=1024,
    	ff1_dropout=0.2,
    	n_head=4,
    	mha_dropout=0.2,
   	kernel_size=3,
    	conv_dropout=0.2,
    	ff2_hsize=1024,
    	ff2_dropout=0.2,
	batch_size=32,
	max_seq_length=512,
	window_size=128,
	decay=0.999,
	kmeans_dropout=0,
	is_left_to_right=False,
	is_share_qk=False,
	use_kmeans_mha=True
)
net.eval()

data = torch.randn(32, 512, 256) # (Batch, Length, Dim)
out = net(data) # (Batch, Length, Dim)
print(out.shape)
# torch.Size([32, 512, 256])
```



## References

- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

- [ESPNet; PR No.2144](https://github.com/espnet/espnet/pull/2144)
- [transformers; modeling_transfo_xl.py](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_transfo_xl.py)

- [Routing Transformer paper](https://arxiv.org/abs/2003.05997)
- [Routing Transformer pytorch implementation](https://github.com/lucidrains/routing-transformer)


## Author

Masao Someki ([@Masao-Someki](htps://github.com/Masao-Someki))

e-mail: `masao.someki@outlook.jp`
