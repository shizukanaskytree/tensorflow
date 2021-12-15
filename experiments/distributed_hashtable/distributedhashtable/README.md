# How to build the package?

```
(hm) wxf@seir19:~/tf2/tensorflow/experiments/distributed_hashtable$ pip install -e .
Obtaining file:///home/wxf/tf2/tensorflow/experiments/distributed_hashtable
  Preparing metadata (setup.py) ... done
Installing collected packages: distributed-hashtable
  Running setup.py develop for distributed-hashtable
Successfully installed distributed-hashtable-0.0.1
```

Since https://stackoverflow.com/questions/6323860/sibling-package-imports

"pip install --editable ./" vs "python setup.py develop"
https://stackoverflow.com/questions/30306099/pip-install-editable-vs-python-setup-py-develop


```
(hm) wxf@seir19:~/tf2/tensorflow/experiments/distributed_hashtable$ pip uninstall distributed_hashtable
```

```
python setup.py develop
```

# Only example3 works

How to run?

1. go to folder example3

```
bash exec_dht_get.sh
```

2. 

```
bash exec_dht_store.sh
```

schema: 

```
{'ip0': ValueWithExpiration(value='192.168.0.1', expiration_time=1639596407.8455963), 'ip1': ValueWithExpiration(value='192.168.0.2', expiration_time=1639596421.6019454), 'ip2': ValueWithExpiration(value='DISABLE', expiration_time=1639596484.9601843)}
```
