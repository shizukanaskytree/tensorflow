# Server

需要添加一个函数能让新加入的 server 被 cluster 接纳.
目前总是先定义一个 cluster 然后我感觉再有新的 server 进来的话就无法再加入了.
这个非常死, 我觉得需要能再让新的机器组建.

相关的代码: 

```
experiments/param_server/in_process_param_server/main.py

tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server_doc.cc
```
