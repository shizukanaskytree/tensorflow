# Tutorial of how to run

https://github.com/learning-at-home/hivemind/blob/master/docs/user/moe.md


线索: hidden_dim 到代码 -> `examples/albert/run_trainer.py` -> `Server.create` -> Code 的部分:

```
        sample_input = name_to_input[expert_cls](3, hidden_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
        else:
            args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)
```


github of hivemind code

```
https://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@github.com/shizukanasky/netmind_hivemind.git
```
