import crypten
import torch
from transformers import BertForSequenceClassification, BertConfig

crypten.cfg.debug.report_cost = True


class BertTinyConfig(BertConfig):
    def __init__(self):
        super().__init__(
            vocab_size=30522,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            num_labels=2
        )


class BertBaseConfig(BertConfig):
    def __init__(self):
        super().__init__()


class BertLargeConfig(BertConfig):
    def __init__(self):
        super().__init__(
            vocab_size=30522,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=512,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            num_labels=2
        )


@crypten.mpc.context.run_multiprocess(2)
def test_bert(config: BertConfig = BertTinyConfig(), input_shape: tuple = (1, 128), device: str = "cuda"):
    crypten.init()

    print(config.__class__.__name__)
    model = BertForSequenceClassification(config)
    model = model.to(device)
    model = model.eval()
    # BertTiny:
    #   TTP: comm byte: 0.48 GB, round: 1180
    #   TFP: comm byte: 0.37 GB, round: 296
    # BertBase:
    #   TTP: comm byte: 13.41 GB, round: 5980
    #   TFP: comm byte: 10.48 GB, round: 1496
    # BertLarge:
    #   TFP: comm byte: 28.49 GB, round: 2936
    #   TTP: comm byte: 36.26 GB, round: 11744
    ct_model = crypten.nn.from_pytorch(model, (
        torch.zeros(input_shape, dtype=torch.int64).to(device),
        torch.zeros(input_shape, dtype=torch.int64).to(device),
        torch.zeros(input_shape, dtype=torch.int64).to(device),
    )).encrypt()
    with torch.no_grad():
        print(f"[rank {crypten.communicator.get().rank}]", "input shape: ", input_shape)
        ct_input_ids = crypten.cryptensor(torch.zeros(input_shape, dtype=torch.int64).to(device))
        ct_attention_mask = crypten.cryptensor(torch.zeros(input_shape, dtype=torch.int64).to(device))
        ct_token_type_ids = crypten.cryptensor(torch.zeros(input_shape, dtype=torch.int64).to(device))

        get_v = ct_model(ct_input_ids, ct_attention_mask, ct_token_type_ids)

        get_v = get_v.get_plain_text()
        need_v = model.forward(torch.zeros(input_shape, dtype=torch.int64).to(device),
                               torch.zeros(input_shape, dtype=torch.int64).to(device),
                               torch.zeros(input_shape, dtype=torch.int64).to(device),
                               )
        print("ct model results", get_v)
        print("pt model results", need_v)


if __name__ == "__main__":
    torch.manual_seed(101)
    torch.random.manual_seed(101)
    torch.cuda.manual_seed(101)
    torch.cuda.random.manual_seed(101)
    crypten.cfg.mpc.provider = "TTP"
    test_bert(BertTinyConfig(), (1, 128), "cuda")
    test_bert(BertBaseConfig(), (1, 128), "cuda")
    test_bert(BertLargeConfig(), (1, 128), "cuda")
