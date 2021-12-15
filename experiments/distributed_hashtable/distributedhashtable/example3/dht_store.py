import distributedhashtable
from distributedhashtable.dht import DHT
import utils

from transformers import HfArgumentParser

from dataclasses import dataclass, field
from typing import List, Optional
from arguments import BaseTrainingArguments
import time 
get_dht_time = time.time

@dataclass
class CollaborationArguments(BaseTrainingArguments):
    statistics_expiration: float = field(
        default=600, metadata={"help": "Statistics will be removed if not updated in this many seconds"}
    )
    backup_every_steps: int = field(
        default=10, metadata={"help": "Frequency of backups to restore from in case of encountering NaN values"}
    )


parser = HfArgumentParser(CollaborationArguments)
(collaboration_args,) = parser.parse_args_into_dataclasses()

validators, local_public_key = utils.make_validators(collaboration_args.experiment_prefix)

print("", collaboration_args.initial_peers)

dht = DHT(
    start=True,
    initial_peers=collaboration_args.initial_peers,
    # client_mode=collaboration_args.client_mode,
    # record_validators=validators,
    # use_ipfs=collaboration_args.use_ipfs,
    # host_maddrs=collaboration_args.host_maddrs,
    # announce_maddrs=collaboration_args.announce_maddrs,
    # identity_path=collaboration_args.identity_path,
)
utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=collaboration_args.use_ipfs)

# pass
# dht.store("key1", "192.168.1.164", expiration_time=get_dht_time() + 300) 

# terminal 0
# dht.store(key="cluster", subkey="ip0", value="192.168.0.1", expiration_time=get_dht_time() + 300)

# terminal 1
# dht.store(key="cluster", subkey="ip1", value="192.168.0.2", expiration_time=get_dht_time() + 300)

# terminal 2
# dht.store(key="cluster", subkey="ip2", value="192.168.0.3", expiration_time=get_dht_time() + 300)

# exe cmd 
dht.store(key="cluster", subkey="ip2", value="DISABLE", expiration_time=get_dht_time() + 300)
# 打印的结果是:
# 因此, 结果可以被覆盖.
# {'ip0': ValueWithExpiration(value='192.168.0.1', expiration_time=1639596407.8455963), 'ip1': ValueWithExpiration(value='192.168.0.2', expiration_time=1639596421.6019454), 'ip2': ValueWithExpiration(value='DISABLE', expiration_time=1639596484.9601843)}




# https://github.com/openai/gym/issues/893#issuecomment-416097416
try:
    del dht
except ImportError:
    pass