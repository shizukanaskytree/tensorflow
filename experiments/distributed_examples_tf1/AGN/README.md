# AGN (Accumulated Gradient Normalization)

This method was formerly known as ADAG (Asynchronous Distributed Adaptive Gradients).

Similar to DOWNPOUR expect that it uses a communications window *T* and accumulates gradients for *T* steps before sending updates to the parameter server.

# How to run?

Terminal 1:
```
python AGN.py --job_name "ps" --task_index 0
```

Terminal 2:
```
python AGN.py --job_name "worker" --task_index 0
```

Terminal 3:
```
python AGN.py --job_name "worker" --task_index 1
```