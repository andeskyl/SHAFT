import time
import torch
import crypten

crypten.cfg.communicator.verbose = True


def test_gelu(runs: int = 10, device: str = "cpu"):
    gelu_approximate = "none"

    x = torch.arange(-5, 5, 0.001)
    y_original = torch.nn.functional.gelu(x, approximate=gelu_approximate)
    y_actual = crypten.cryptensor(x).gelu(approximate=gelu_approximate).get_plain_text()
    max_err = (y_original - y_actual).abs().max()
    min_err = (y_original - y_actual).abs().min()
    avg_err = (y_original - y_actual).abs().mean()

    size = (128, 3072)
    x = crypten.cryptensor(torch.zeros(size), device=device)
    crypten.reset_communication_stats()
    start_time = time.time()

    for _ in range(runs):
        x.gelu(approximate=gelu_approximate)
    comm_time = time.time() - start_time
    stats = crypten.get_communication_stats()
    comm_bytes = stats["bytes"]
    comm_rounds = stats["rounds"]

    log = f"gelu: max error: {max_err}, avg error: {avg_err}, min error: {min_err} "
    log += f"{size} time: {comm_time / runs}s, bytes: {comm_bytes / (2 ** 20) / runs} MB, rounds: {comm_rounds / runs}"
    log = f"[rank {crypten.comm.get().get_rank()}] [provider {crypten.cfg.mpc.provider}] " + log
    print(log)


def test_softmax(runs: int = 10, device: str = "cpu"):
    x = torch.arange(-5, 5, 0.001)
    y_original = torch.nn.functional.softmax(x, -1)
    y_actual = crypten.cryptensor(x).softmax(-1).get_plain_text()
    max_err = (y_original - y_actual).abs().max()
    avg_err = (y_original - y_actual).abs().mean()
    min_err = (y_original - y_actual).abs().min()

    size = (12, 128, 128)
    x = crypten.cryptensor(torch.zeros(size), device=device)
    crypten.reset_communication_stats()
    start_time = time.time()

    for _ in range(runs):
        y = x.softmax(-1)
    comm_time = time.time() - start_time
    stats = crypten.get_communication_stats()
    comm_bytes = stats["bytes"]
    comm_rounds = stats["rounds"]

    log = f"softmax: max error: {max_err}, avg error: {avg_err}, min error: {min_err} "
    log += f"{size} time: {comm_time / runs}s, bytes: {comm_bytes / (2 ** 20) / runs} MB, rounds: {comm_rounds / runs}"
    log = f"[rank {crypten.comm.get().get_rank()}] [provider {crypten.cfg.mpc.provider}] " + log
    print(log)


def test_embedding(runs: int = 10, device: str = "cuda"):
    num_embeddings = 32
    embedding_dim = 768

    ct_emb = crypten.nn.module.Embedding().to(device)
    pt_emb = torch.nn.Embedding(num_embeddings, embedding_dim, device=device)
    pt_x = torch.randint(0, num_embeddings, (1, 128), device=device)
    y_original = pt_emb.forward(pt_x)
    y_actual = ct_emb.forward(
        (crypten.cryptensor(pt_emb.weight), crypten.cryptensor(pt_x), None)
    ).get_plain_text()
    max_err = (y_original - y_actual).abs().max()
    avg_err = (y_original - y_actual).abs().mean()
    min_err = (y_original - y_actual).abs().min()
    print(f"[rank {crypten.comm.get().get_rank()}] [provider {crypten.cfg.mpc.provider}] "
          f"Embedding max error: {max_err:.4f}, avg error: {avg_err:.6f}, min error: {min_err:.6f}")
    embedding_time = {}
    embedding_bytes = {}
    embedding_rounds = {}

    embedding_sizes = [(1, 128), (1, 256)]
    pt_emb = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)
    for embedding_size in embedding_sizes:
        embedding_in = crypten.cryptensor(torch.randint(0, num_embeddings, embedding_size), device=device)
        crypten.reset_communication_stats()
        start_time = time.time()

        for _ in range(runs):
            ct_emb.forward(
                (crypten.cryptensor(pt_emb.weight), embedding_in, None)
            )
        embedding_time[embedding_size[1]] = time.time() - start_time
        stats = crypten.get_communication_stats()
        embedding_bytes[embedding_size[1]] = stats["bytes"]
        embedding_rounds[embedding_size[1]] = stats["rounds"]

    for embedding_size in embedding_sizes:
        print(f"[rank {crypten.comm.get().get_rank()}] [provider {crypten.cfg.mpc.provider}] "
              f"Embedding ({embedding_size[0]}, {embedding_size[1]}) "
              f"time: {embedding_time[embedding_size[1]] / runs:.4f}s, "
              f"bytes: {embedding_bytes[embedding_size[1]] / 2 ** 20 / runs:.0f} MB, "
              f"rounds: {embedding_rounds[embedding_size[1]] / runs:.0f}"
              )


@crypten.mpc.context.run_multiprocess(2)
def main(device: str = "cuda"):
    test_gelu(device=device)
    test_softmax(device=device)
    test_embedding(device=device)


if __name__ == "__main__":
    torch.manual_seed(101)
    torch.random.manual_seed(101)
    torch.cuda.manual_seed(101)
    torch.cuda.random.manual_seed(101)
    crypten.cfg.mpc.provider = "TTP"
    main()
    crypten.cfg.mpc.provider = "TFP"
    main()
