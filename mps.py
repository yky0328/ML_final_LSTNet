import torch
import numpy as np
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1)
    print(type(x))
    x = x.to('mps')
    print (x)
else:
    print ("MPS device not found.")
