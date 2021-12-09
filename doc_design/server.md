# Server

需要添加一个函数能让新加入的 server 被 cluster 接纳.
目前总是先定义一个 cluster 然后我感觉再有新的 server 进来的话就无法再加入了.
这个非常死, 我觉得需要能再让新的机器组建.

相关的代码: 

```
experiments/param_server/in_process_param_server/main.py

tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server_doc.cc
```

Server and device id are paired. 

What is a server?


```
self._server = c_api.TF_NewServer(self._server_def.SerializeToString())
```
- the only parameter passed to server is
  - server_def

所以也是很简洁的. 目前我推进的部分还有简洁. 或者我应该规定.
我目前的参数是子集: `self._server_def`

```
    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_spec.as_cluster_def(),
        job_name=job_name,
        task_index=task_index,
        protocol=protocol)
```


```
    server_def = tensorflow_server_pb2.ServerDef(
        # del # cluster=cluster_spec.as_cluster_def(),
        job_name=job_name,
        task_index=task_index,
        protocol=protocol)
```

参数规定: 

job_name = "worker" or "ps"
task_index = 0, 1, 2, 未必连续, 因为总有会挂掉的, 只能序列增加, 增长, 就像 pid.


# I need a global ID

https://medium.com/@sandeep4.verma/system-design-distributed-global-unique-id-generation-d6a440cc8e5

I need dht id.

class DHTID(int) in hivemind/dht/routing.py

DHTID => task_idx => or I say task id.


The advantage is:
- If you have a DHT ID, then you are doomed to be identified in DHT!


目的地:
- 这里不是你的终点

条件:
- 物质生存条件不满足, 随时会断粮和精神折磨
- 能力不满足, tf code 95% 都不懂


只有条件满足了才能推入下一步, 否则就是大跃进.
历史告诉我们, 大跃进会造成打饥荒和灾难.

还有就是这个也是运行时, 想法也在改变.

现阶段只能做一个 toy.
hivemind 的代码量还是可以控制的, 单人
但是超过以后就不行了.

前面是谁, ip
后面是谁, ip
[ ]--{ }--[ ] | [ ]--{ }--[ ]

认为切割的 block. 
前后关联.

4-16 GPUs 来测试就不错了.

切割 10, nlp 模型相关的

coordinator monitors that 
the sequence is 1-2-3-4-5-6-7-8-9-10 

1-2-3||-4-5-6||-7-8-9||-10
  1      2        3     4  <= 4 GPUs

1-2|-3-4|-5-6|-7-8|-9-|10
                          <= 5 GPUs

1-2-3-4-5-6-7-8-9-10
  \-3-/
  \-3-/
  \-3-/
   avg



fault tolerance

checkpoint upload

every op as a function 
1-10 => 1-N

论文不 appreciate your implementation, but your idea. 
- dynamic changing the graph and assignment - research 
  - since coordinator can monitor metric 需要设计
  - 公式
    - throughput
    - time of some bottleneck




所有的都是逻辑和数学, 满足条件就能运行.

把模型手动切分成 10 份的条件满足. 
- 编号化每个模块
- mapping to function name with args so that we can call them

第一步统计, 知道系统里有几个 server, GPUs
- dht 的模块可以统计, 参考 albert/train_monitor.py
- 首先需要执行的是 coordinator.py, 然后它提供了其他 peer 的 init_peers 这个参数供接入.
- coordinator.py 是全知全能的神.
- coordinator 能够知道 其他 peers 发给它的信息
- coordinator 也能够发送给 其他 peers 信息供他们执行图和链接彼此, 前驱后继.
- 总是要先执行 coordinator_keep_query.py 用来实时 monitor cluster 的动态, 获取信息.
- 可以用 local server, 多 process, 多个 port 来模拟多个 dht, dht 对应一个 server



如果有 4 台机器, 每个机器 1 个 GPU, 通过 coordinator 的方式把他们串起来
- 这个过程默认是足够稳定的.
- 初始化启动过程
  - iter all 10 blocks so that we can compose the, 以 blocks 为主体
    - assign some blocks to server_ID:GPU_ID, 
      - server 待定的参数包括 blocks_ID to execute
      - set or map
      - DHTID sort by large or small order
    - 2 vars: forward to ip, backward to ip for each server
    - server 待定的参数还包括 forward ip, backward ip
  - 这些满足以后可以开始执行训练


- 节点加入, coordinator 需要重新编排
  - 如果是做数据拆分, batch size 拆分, 一个 server 要给两个 server 传 partitioned 的 tensor
  - 条件不满足
    - grpc 调用 2 个 servers 的 forward 函数, 发出后是 blocking 的状态
      - 因为 grpc 是 end to end 的操作, 所以数据切分不可行.
      - 如果要可行, 那么就建立两个 grpc 通道, 得到的结果也要合并一下, batch size 那个维度的合并
    - 中间发出后没有返回值
    - grpc 有 async 的, 需要尝试. 待定, 未知, 下一的步骤都未知.
    - forward, backward 的实现因为参数的改变有变化
      - forward 的 input schema, input 都会变化
      - backward 的 backward schema
  -  partition 后需要整合, i.e., reduce op 怎么做?
    - 条件不具备

  - 按照顺序有限供给前方 server id.
  - forward input partition
  - backward input partition 

- 节点加入, coordinator 需要重新编排
  - 

dht 的节点加入
- json 提供所有的 server ip 信息
- 

hivemind.moe.server
- a sever hosts all models partitions
  - https://learning-at-home.readthedocs.io/en/latest/modules/server.html
  - A hivemind server hosts one or several experts and processes incoming requests to those experts.


如果有 4 台机器, 每个机器 2 个 GPU, 通过 coordinator 的方式把他们串起来

- remote expert 每个机器都有 N 份全部切分完毕, 随时准备被的 schedule.
  - coordinator spread the execution flow
  - subgraph executor follows the instructions to do forward and backward.
  - 



END.