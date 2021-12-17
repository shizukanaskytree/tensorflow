from atom.runtime import Runtime
from atom.model_backend import ModelBackend

model_inputs = DUMMY_TENSORS

model_backends = {'model_backends': ModelBackend(**kwargs)}

runtime = Runtime(model_backends)
runtime.start()  # start runtime in background thread. To start in current thread, use runtime.run()
runtime.ready.wait()  # await for runtime to load all experts on device and create request pools

# Runtime 已经开始
# ========================================================


future = runtime.expert_backends['expert_name'].forward_pool.submit_task(*model_inputs)
print("Returned:", future.result())
runtime.shutdown()
