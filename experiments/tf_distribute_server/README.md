Note:

https://www.notion.so/xiaofengwu/model-parallelism-in-TF2-8e4009f69dd5422fa911da216fb55d9e

不成功, 不知道为什么.


```
python example.py --job_name="ps" --task_index=0
CUDA_VISIBLE_DEVICES=0 python example.py --job_name="worker" --task_index=0
CUDA_VISIBLE_DEVICES=1 python example.py --job_name="worker" --task_index=1
CUDA_VISIBLE_DEVICES=2 python example.py --job_name="worker" --task_index=2
```
https://github.com/nottombrown/distributed-tensorflow-example