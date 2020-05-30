import torch

def LogGPU():
    print(">>>>>  GPU info:")
    if torch.cuda.is_available():
        print("    CUDA is available")
    else:
        print("    CUDA is not available")

    gpu_count = torch.cuda.device_count()
    for i in range(0, gpu_count):
        print("    GPU %d: %s" % (i, torch.cuda.get_device_name(i)))

    gpu_current = torch.cuda.current_device()
    print("    Current GPU: %s" % torch.cuda.get_device_name(gpu_current))


LogGPU()
