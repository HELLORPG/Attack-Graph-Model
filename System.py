"""
本文件用于测试当前的环境。
1. 查看Torch配置是否正确。
2. 查看当前的GPU配置是否符合预期。
"""

import torch
# import tensorflow


def LogPyTorchGPU():
    print(">>>>>  PyTorch GPU info:")
    if torch.cuda.is_available():
        print("    CUDA is available")
    else:
        print("    CUDA is not available")

    gpu_count = torch.cuda.device_count()
    for i in range(0, gpu_count):
        print("    GPU %d: %s" % (i, torch.cuda.get_device_name(i)))

    gpu_current = torch.cuda.current_device()
    print("    Current GPU: %s" % torch.cuda.get_device_name(gpu_current))


def LogPyTorch():
    print(">>>>>  PyTorch info:")
    print("    version: %s" % torch.__version__)


# def LogTensorflow():
#     print(">>>>>  Tensorflow info:")
#     print("    version: %s" % tensorflow.__version__)


def main():
    LogPyTorch()
    LogPyTorchGPU()

    # LogTensorflow()


main()



