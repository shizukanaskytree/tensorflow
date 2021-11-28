# Multi-worker configuration
# A cluster with jobs and tasks
# 
# `cluster`
#   several jobs
#       multi  `task` s

# TF_CONFIG is a JSON string used to specify the cluster configuration 
# for each worker that is part of the cluster.

# `TF_CONFIG`
# 'worker' or 'chief'
# 

tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}