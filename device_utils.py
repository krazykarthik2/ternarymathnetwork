"""
Hardware device abstraction utility.
Seamlessly detects CUDA, DirectML (if functional), or falls back to optimized CPU.
Ensures zero code changes when migrating to a GPU/CUDA VM.
"""
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")

def get_device_info():
    dev = get_device()
    info = {"device": str(dev), "type": dev.type if hasattr(dev, "type") else str(dev)}
    if dev.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["capability"] = torch.cuda.get_device_capability(0)
    elif "privateuseone" in str(dev).lower() or "directml" in str(dev).lower():
        info["name"] = "DirectML iGPU/Accelerator"
    else:
        info["name"] = f"CPU ({torch.get_num_threads()} threads)"
    return info

if __name__ == "__main__":
    print(f"Device configuration: {get_device_info()}")
