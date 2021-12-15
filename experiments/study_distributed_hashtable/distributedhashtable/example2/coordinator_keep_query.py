# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

import multiprocessing as mp
import asyncio
import uvloop

from transformers import HfArgumentParser
from typing import List, Optional
from dataclasses import asdict, dataclass, field

import distributedhashtable 
from distributedhashtable.dht import DHT
from distributedhashtable.utils.logging import get_logger, use_hivemind_log_handler
import utils

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


@dataclass
class BaseTrainingArguments:
    coordinator_name: str = field(
        metadata={"help": "A unique 'name' of this loose cluster, used to store metadata on the DHT"}
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
class TrainingMonitorArguments(BaseTrainingArguments):
    """
    Note: You might want to have several initial peers so that if one dies,
    new workers still can join the collaboration via alive initial peers' addresses.
    Specify initial_peers argument for that purpose
    """

    use_google_dns: bool = field(
        default=False,
        metadata={
            "help": "Use Google DNS to determine the public IP address of this machine (and add it to --announce_maddrs)"
        },
    )
    refresh_period: float = field(default=30, metadata={"help": "Period (in seconds) for fetching the keys from DHT"})
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Name of Weights & Biases project to report the training progress to"}
    )
    save_checkpoint_step_interval: int = field(
        default=5, metadata={"help": "Frequency (in steps) of fetching and saving state from peers"}
    )
    model_config_path: str = field(
        default="https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
        metadata={"help": "Path to the model config"},
    )
    repo_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local repository to store the model and optimizer states"}
    )
    repo_url: Optional[str] = field(
        default=None, metadata={"help": "URL of Hugging Face Hub repository to upload the model and optimizer states"}
    )
    upload_interval: Optional[float] = field(
        default=None, metadata={"help": "Frequency (in seconds) of uploading the model to Hub"}
    )
    store_checkpoins: bool = field(default=False, metadata={"help": "If True, enables CheckpointHandler"})


parser = HfArgumentParser(TrainingMonitorArguments)
coordinator_args, = parser.parse_args_into_dataclasses()

coordinator_name = coordinator_args.coordinator_name
validators, local_public_key = utils.make_validators(coordinator_name)


def switch_to_uvloop() -> asyncio.AbstractEventLoop:
    """
    stop any running event loops; install uvloop; then create, set and return a new event loop
    """
    try:
        asyncio.get_event_loop().stop()  # if we're in jupyter, get rid of its built-in event loop
    except RuntimeError as error_no_event_loop:
        pass  # this allows running DHT from background threads with no event loop
    uvloop.install()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


class CoordinatorProcessing(mp.Process):
    def __init__(self):
        super().__init__()
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)
        
        # 有多少个参数?
        # ==========
        # DHT 有 8 个.
        # DHTNode.create 25 个, 4 个和 DHT 相同
        self.dht = DHT(
            start=True,
            initial_peers=coordinator_args.initial_peers,
            record_validators=validators,
            use_ipfs=coordinator_args.use_ipfs,
            host_maddrs=coordinator_args.host_maddrs,
            announce_maddrs=coordinator_args.announce_maddrs,
            identity_path=coordinator_args.identity_path,
        )
        utils.log_visible_maddrs(self.dht.get_visible_maddrs(), only_p2p=coordinator_args.use_ipfs)
        self.start()


    def run(self) -> None:
        loop = switch_to_uvloop()
        async def _run():
            # keep querying to get global information
            while True:
                # metrics_dict = dht.get("store_test_key", latest=True)
                recv = self.dht.get("key1")
                if recv is not None:
                    print(recv.value) #  == "value1"
                    print(recv)
                    # peer_info = [recv[peer].value for peer in recv]
                    # print(peer_info)
                    # break

        loop.run_until_complete(_run())


    def send_msg(self):
        print("sending")
        self._outer_pipe.send(("_shutdown", ["args"], {"kwargs_0": 0}))
        print("sent")


    def recv_msg(self):
        """Start automatically"""
        print("start recving")

        while True:
            method, args, kwargs = self._inner_pipe.recv()
            if method is not None:
                print(method, args, kwargs)


coordinator = CoordinatorProcessing()
