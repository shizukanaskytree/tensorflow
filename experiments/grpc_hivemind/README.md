# Description

In gRPC, a client application can directly call a method on a server application on a different machine as if it were a local object, making it easier for you to create distributed applications and services.

slides:
https://docs.google.com/presentation/d/1A_jk4GBlpX7IJg0rTcuJKHVtkK1kibypU3swc9cTrUo/edit#slide=id.g107317a3b2c_0_74

notes:
https://www.notion.so/xiaofengwu/Atom-Volunteer-Computing-79cc2b7d9a2e40099e0cd7c66a485905

# How to run?

Terminal 1: start the server

```
wxf@seir19:~/tf2/tensorflow/experiments/grpc_hivemind$ python server.py 
forward-ing
forward-ed
backward-ing
backward-ed
info-ing
info-ed
```

Terminal 2: call rpc from the client side

```
(hm) wxf@seir19:~/tf2/tensorflow/experiments/grpc_hivemind$ python client.py 
```

# 测试 utils/grpc.py

timed storage => grpc cache, ChannelCache