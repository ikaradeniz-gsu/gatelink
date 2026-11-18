"""Pick the best available accelerator: CUDA (NVIDIA), MPS (Apple Silicon) or CPU."""


def best_device(requested: str = "auto") -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
