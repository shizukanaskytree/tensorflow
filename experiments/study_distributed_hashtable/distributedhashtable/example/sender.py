from dataclasses import dataclass
from distributedhashtable.dht import DHT
from distributedhashtable.utils.logging import get_logger, use_hivemind_log_handler

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

from transformers import HfArgumentParser, TrainingArguments

@dataclass
class SenderArgs(TrainingArguments):
    peers_endpoint: str = ""


parser = HfArgumentParser(SenderArgs)
sender_args, = parser.parse_args_into_dataclasses()

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

initial_peers = [sender_args.peers_endpoint]
print(initial_peers)
dht_node = DHT(initial_peers=initial_peers, start=True)

# visible_maddrs_str = [str(a) for a in dht_node.get_visible_maddrs()]
# logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

dht_node.store(key="dht_key", value="dht_store_value", expiration_time=6)
