import gc
import torch

def clear_gpu_memory(**kwargs):
    print("--- Starting GPU Memory Cleanup ---")
    if kwargs:
        for name, object in kwargs.items():
            try:
                del object
                print(f"Deleted object: {name}")
            except Exception as e:
                print(f"Could not delete object {name}: {e}")
    gc.collect()
    if torch.cuda.is_available():
        if torch.cuda.memory_allocated() > 0 or torch.cuda.memory_reserved() > 0:
            torch.cuda.empty_cache()
            print("Emptied CUDA cache.")
        else:
            print("No GPU memory to clear.")
    else:
        print("CUDA is not available.")
    print("--- GPU Memory Cleanup Completed ---")