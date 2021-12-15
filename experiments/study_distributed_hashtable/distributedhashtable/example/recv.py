from distributedhashtable.dht import DHT
from distributedhashtable.utils.logging import get_logger, use_hivemind_log_handler

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

initial_peers = None
dht_node = DHT(initial_peers=initial_peers, start=True)

visible_maddrs_str = [str(a) for a in dht_node.get_visible_maddrs()]
logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

while True:
    dht_recv = dht_node.get("dht_key", latest=True)
    if dht_recv is not None:
        print('---')
        dht_recv = dht_recv.value
        print(f'dht_recv: {dht_recv}')

