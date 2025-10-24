import torch

def check_gpu():
    print("PyTorch version:", torch.__version__)
    print("\nCUDA available:", torch.cuda.is_available())
    print("XPU (AMD GPU) available:", hasattr(torch, 'xpu') and torch.xpu.is_available())
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("\nAMD GPU Information:")
        print("Device name:", torch.xpu.get_device_name())
        print("Device count:", torch.xpu.device_count())
        
        # Test tensor creation on AMD GPU
        try:
            x = torch.randn(3, 3).xpu()
            print("\nSuccessfully created tensor on AMD GPU:")
            print(x)
        except Exception as e:
            print("\nError creating tensor on AMD GPU:", str(e))
    else:
        print("\nNo AMD GPU available. The system will use CPU for computations.")

if __name__ == "__main__":
    check_gpu()