import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

def main():
    crypten.init()
    device = "cuda"
    runs = 10
    gelu_time, gelu_bytes, gelu_rounds = {}, {}, {}
    approximate = "none"

    x = torch.arange(-5, 5, 0.001)
    y_original = torch.nn.functional.gelu(x, approximate=approximate)
    y_actual = crypten.cryptensor(x).gelu(approximate=approximate).get_plain_text()
    max_err = (y_original - y_actual).abs().max()
    avg_err = (y_original - y_actual).abs().mean()
    
    for gelu_size in [(128, 3072), (128, 4096)]:
        gelu_in = crypten.cryptensor(torch.zeros(gelu_size), device=device)
        crypten.reset_communication_stats()
        start_time = time.time()
        
        for _ in range(runs):
            gelu_in.gelu(approximate=approximate)
        gelu_time[gelu_size[1]] = time.time() - start_time
        stats = crypten.get_communication_stats()
        gelu_bytes[gelu_size[1]] = stats["bytes"]
        gelu_rounds[gelu_size[1]] = stats["rounds"]
    
    if crypten.comm.get().get_rank() == 0:
        print(f"max error: {max_err:.4f}, avg error: {avg_err:.6f}")
        for gelu_size in [[128, 3072], [128, 4096]]:
            print(f"({gelu_size[0]}, {gelu_size[1]}) "
                f"time: {gelu_time[gelu_size[1]] / runs:.4f}s, "
                f"bytes: {gelu_bytes[gelu_size[1]] / 1048576 / runs:.0f} MB, "
                f"rounds: {gelu_rounds[gelu_size[1]] / runs:.0f}"
            )

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()