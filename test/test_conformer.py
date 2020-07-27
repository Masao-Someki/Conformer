# This is a test script

import torch
import json
from CF import get_conformer

def test_conformer():
    conf = json.load(open('conformer.conf'))
    net = get_conformer(**conf)
    net.eval()
    data = torch.randn(1, 32, conf['d_model'])
    # data should be formatted as (B, L, D)
    # B as batch-size, L as sequence-length, D as feature-dimension.
    out = net(data)
    assert out.shape == (1, 32, conf['d_model'])
    
