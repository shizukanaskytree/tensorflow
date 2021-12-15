YOUR_EXPERIMENT_NAME=experiment_dht
ADDR=/ip4/192.168.0.19/tcp/36195/p2p/QmcVBVMmE5krpBtAP91DV3SPTMEUGxz2yL6keGVEbQ2JsV
python dht_store.py --experiment_prefix ${YOUR_EXPERIMENT_NAME} --initial_peers ${ADDR} 
