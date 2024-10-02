import time
import torch
import crypten
from multiprocess_launcher import MultiProcessLauncher

def main():
    crypten.init()
    device = "cuda"
    runs = 10
    softmax_time, softmax_bytes, softmax_rounds = {}, {}, {}

    # softmax test
    for softmax_l in [32, 64, 128, 256]:
        softmax_in = crypten.cryptensor(torch.zeros([softmax_l]), device=device)
        crypten.reset_communication_stats()
        start_time = time.time()
        for _ in range(runs):
            softmax_in.softmax(-1)
        softmax_time[softmax_l] = time.time() - start_time
        stats = crypten.get_communication_stats()
        softmax_bytes[softmax_l] = stats["bytes"]
        softmax_rounds[softmax_l]  = stats["rounds"]

    if crypten.comm.get().get_rank() == 0:
        for softmax_l in [32, 64, 128, 256]:
            print(f"l={softmax_l} "
                  f"time: {softmax_time[softmax_l] / runs:.4f}s, "
                  f"bytes: {softmax_bytes[softmax_l] / 1048576 / runs:.4f} MB, "
                  f"rounds: {softmax_rounds[softmax_l] / runs:.0f}")

if __name__ == "__main__":
    launcher = MultiProcessLauncher(2, main)
    launcher.start()
    launcher.join()
    launcher.terminate()