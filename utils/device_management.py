import torch
from typing import List, Optional

def available_devices() -> List[torch.device]:
    devs: List[torch.device] = []
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    devs.append(torch.device("cpu"))
    return devs

