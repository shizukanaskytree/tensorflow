COORDINATOR=experiment_dht
ADDR=/ip4/192.168.0.19/tcp/42021/p2p/QmVMDUhPCSWkicyvW746TFJXr72jKcw5mFn5gs5tt5joUr
python coordinator_set_dht_key_value.py --coordinator_name ${COORDINATOR} --initial_peers ${ADDR}
