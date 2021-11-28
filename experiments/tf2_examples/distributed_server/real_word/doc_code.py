"""
Purpose
- learn the code



Clusters in the real world

Keyword:

* TF_CONFIG
* multi-worker setup
* the **standard** way in TensorFlow to specify the cluster configuration 
"""

"""
In a real production environment, you will run all tasks in different processes on different machines. 
The simplest way to configure cluster information on each task is to set "TF_CONFIG" environment variables 
and use a tf.distribute.cluster_resolver.TFConfigClusterResolver to parse "TF_CONFIG".

For a general description about "TF_CONFIG" environment variables, refer to the Distributed training guide.

If you start your training tasks using Kubernetes or other configuration templates, 
it is very likely that these templates have already set “TF_CONFIG" for you.

-----------------------------------------------------------------------------------------------------

from https://www.tensorflow.org/guide/distributed_training#setting_up_tf_config_environment_variable

One of the key differences to get multi worker training going, as compared to multi-GPU training, is the multi-worker setup. 
The 'TF_CONFIG' environment variable is the **standard** way in TensorFlow to specify the cluster configuration to each worker 
that is part of the cluster. Learn more in the setting up TF_CONFIG section of this document.

For more details about MultiWorkerMirroredStrategy, consider the following tutorials:

=> Multi-worker training with Keras Model.fit
=> Multi-worker training with a custom training loop


setting up TF_CONFIG section
============================
https://www.tensorflow.org/guide/distributed_training#TF_CONFIG
不是很多就一页. 


Setting up the TF_CONFIG environment variable
=============================================

For multi-worker training, as mentioned before, you need to set up the 'TF_CONFIG' environment variable for each binary running in your cluster. 
The 'TF_CONFIG' environment variable is a JSON string which specifies what tasks constitute a cluster, their addresses and each task's role in the cluster. 
The tensorflow/ecosystem repo provides a Kubernetes template, which sets up 'TF_CONFIG' for your training tasks.

There are two components of 'TF_CONFIG': a cluster and a task.

A cluster provides information about the training cluster, which is a dict consisting of different types of jobs such as workers. 
In multi-worker training, there is usually one worker that takes on a little more responsibility like saving checkpoint and writing summary file for TensorBoard in addition to what a regular worker does. Such worker is referred to as the "chief" worker, and it is customary that the worker with index 0 is appointed as the chief worker (in fact this is how tf.distribute.Strategy is implemented).

A task on the other hand provides information about the current task. The first component cluster is the same for all workers, and the second component task is different on each worker and specifies the type and index of that worker.
One example of 'TF_CONFIG' is:

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})

The "TF_CONFIG" of the evaluator can be:

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "evaluator": ["host7:port"]
    },
    "task": {"type": "evaluator", "index": 0}
})

The "cluster" part in the above "TF_CONFIG" string for the evaluator is optional.

===========================================================

If you use the same binary for all tasks

If you prefer to run all these tasks using a single binary, you will need to let your program branch into different roles at the very beginning:

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
if cluster_resolver.task_type in ("worker", "ps"):
  # Start a TensorFlow server and wait.
elif cluster_resolver.task_type == "evaluator":
  # Run side-car evaluation
else:
  # Run the coordinator.


The following code starts a TensorFlow server and waits:

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol=cluster_resolver.rpc_layer or "grpc",
    start=True)
server.join()


tf.distribute.cluster_resolver.TFConfigClusterResolver
https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/TFConfigClusterResolver

"""
