import time
now = time.time

# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

# 怎么用?
# TimedStorage[ChannelInfo, Tuple[Union[grpc.Channel, grpc.aio.Channel], Dict]]

# code 怎么理解?
import timed_storage

# class TimedStorage(Generic[KeyType, ValueType])
# TimedStorage[ChannelInfo, Tuple[Union[grpc.Channel, grpc.aio.Channel], Dict]]

keytype = int 
valuetype = int 

ts = timed_storage.TimedStorage[keytype, valuetype]()

# store
# get
# items
# top
# clear
# 

ts.store(key=1, value=10, expiration_time=now()+30000)

