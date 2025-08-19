# Test For Trust Third Party Provider

## run

```shell
python3 run_test_ttp.py
```

```shell
python3 run_test_bert.py
```

## use

specify in the code

```shell
import crypten

crypten.cfg.mpc.provider = "TTP"
```

or edit the configs/default.yaml

```yaml
mpc:
  provider: "TTP" # default is TFP
```