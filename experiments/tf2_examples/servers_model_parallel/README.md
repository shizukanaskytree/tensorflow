https://medium.com/@willburton_48961/how-to-use-distributed-tensorflow-to-split-your-tensorflow-graph-between-multiple-machines-f48ffca2810c 

note:
https://www.notion.so/xiaofengwu/Model-parallel-TF1-code-multi-server-2b3d2367db0d4f94a26ec7b598b598d9

# Goal

3 servers
- worker 0: subgraph
    - p0: 10.0.0.111 
- worker 1: subgraph
    - p1: 10.0.0.93
- chief: run the whole graph.
    - p2: 10.0.0.25