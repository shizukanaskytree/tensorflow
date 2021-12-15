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


