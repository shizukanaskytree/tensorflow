# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

from transformers import HfArgumentParser
from typing import List, Optional
from dataclasses import asdict, dataclass, field

from distributedhashtable.dht import DHT
from distributedhashtable.utils.logging import get_logger, use_hivemind_log_handler
import utils
from arguments import BaseTrainingArguments
import time

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)

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
monitor_args, = parser.parse_args_into_dataclasses()

experiment_prefix = monitor_args.experiment_prefix
validators, local_public_key = utils.make_validators(experiment_prefix)

# Design input:
# 
# monitor_args.initial_peers
# []

# validators
# [<hivemind.dht.schema...6680ac2b0>, <hivemind.dht.crypto...6680bda00>]

# experiment_prefix
# 'YOUR_EXPERIMENT_NAME'

# monitor_args.use_ipfs
# False

# monitor_args.host_maddrs
# ['/ip4/0.0.0.0/tcp/0']

# monitor_args.announce_maddrs
# []

# monitor_args.identity_path
# None

dht = DHT(
    start=True,
    initial_peers=monitor_args.initial_peers,
    # record_validators=validators,
    # use_ipfs=monitor_args.use_ipfs,
    # host_maddrs=monitor_args.host_maddrs,
    # announce_maddrs=monitor_args.announce_maddrs,
    # identity_path=monitor_args.identity_path,
)

utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=monitor_args.use_ipfs)

while True:
    # metrics_dict = dht.get("store_test_key", latest=True)
    recv = dht.get("cluster")
    if recv is not None:
        # cnt = 0
        for item in recv:
            # cnt += 1
            print(item)
            # {'ip2': ValueWithExpiration(value='192.168.0.3', expiration_time=1639596209.9649987), 'ip1': ValueWithExpiration(value='192.168.0.2', expiration_time=1639596250.6313906), 'ip0': ValueWithExpiration(value='192.168.0.1', expiration_time=1639596295.814378)}
            time.sleep(0.1)
        
        # print(cnt)
            # print(item)
        # print(recv.value) #  == "value1"
        
        # print(recv)
        # peer_info = [recv[peer].value for peer in recv]
        # print(peer_info)
    time.sleep(0.1)
