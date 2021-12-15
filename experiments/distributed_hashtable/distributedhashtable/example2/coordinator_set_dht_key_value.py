from distributedhashtable.dht import DHT
import utils

from transformers import HfArgumentParser

from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class BaseTrainingArguments:
    coordinator_name: str = field(
        metadata={"help": "A unique 'name' of this experiment, used to store metadata on the DHT"}
    )
    initial_peers: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Multiaddrs of the peers that will welcome you into the existing collaboration. "
            "Example: /ip4/203.0.113.1/tcp/31337/p2p/XXXX /ip4/203.0.113.2/tcp/7777/p2p/YYYY"
        },
    )
    use_ipfs: bool = field(
        default=False,
        metadata={
            "help": "Use IPFS to find initial_peers. If enabled, you only need to provide /p2p/XXXX part of the multiaddrs "
            "for the initial_peers (no need to specify a particular IPv4/IPv6 host and port)"
        },
    )
    host_maddrs: List[str] = field(
        default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"],
        metadata={
            "help": "Multiaddrs to listen for external connections from other p2p instances. "
            "Defaults to all IPv4 interfaces and the TCP protocol: /ip4/0.0.0.0/tcp/0"
        },
    )
    announce_maddrs: List[str] = field(
        default_factory=list,
        metadata={"help": "Visible multiaddrs the host announces for external connections from other p2p instances"},
    )
    identity_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a pre-generated private key file. If defined, makes the peer ID deterministic. "
            "May be generated using ``./p2p-keygen`` from ``go-libp2p-daemon``."
        },
    )


@dataclass
class CollaborationArguments(BaseTrainingArguments):
    statistics_expiration: float = field(
        default=600, metadata={"help": "Statistics will be removed if not updated in this many seconds"}
    )
    backup_every_steps: int = field(
        default=10, metadata={"help": "Frequency of backups to restore from in case of encountering NaN values"}
    )


parser = HfArgumentParser(CollaborationArguments)
(coordinator_name,) = parser.parse_args_into_dataclasses()

validators, local_public_key = utils.make_validators(coordinator_name.coordinator_name)

print("coordinator_name.initial_peers: ", coordinator_name.initial_peers)

dht = DHT(
    start=True,
    initial_peers=coordinator_name.initial_peers,
    # client_mode=coordinator_name.client_mode,
    record_validators=validators,
    use_ipfs=coordinator_name.use_ipfs,
    host_maddrs=coordinator_name.host_maddrs,
    announce_maddrs=coordinator_name.announce_maddrs,
    identity_path=coordinator_name.identity_path,
)
utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=coordinator_name.use_ipfs)

dht.store("key1", "192.168.1.162", expiration_time=0.1)
dht.store("key1", "192.168.1.163", expiration_time=0.1)
dht.store("key1", "192.168.1.164", expiration_time=0.1)

## store time, the coordinator
dht.store(key="cluster_info", value="", expiration_time=0.1)

