YOUR_EXPERIMENT_NAME=experiment_dht
ADDR=/ip4/127.0.0.1/tcp/34172/p2p/QmVm9HFMPQtmNH8PyH1iNFWEiY7tWzh1b1yEjSWFWsEJPA
python dht_store.py --experiment_prefix ${YOUR_EXPERIMENT_NAME} --initial_peers ${ADDR} 
