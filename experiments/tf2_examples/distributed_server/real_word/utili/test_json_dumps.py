import json

ret = json.dumps(
    {
        "cluster": {
            "worker": ["host1:port", "host2:port", "host3:port"],
            "ps": ["host4:port", "host5:port"],
        },
        "task": {"type": "worker", "index": 1},
    },
    sort_keys=True,
    indent=4,
)

print(ret)

"""
Output:

{
    "cluster": {
        "ps": [
            "host4:port",
            "host5:port"
        ],
        "worker": [
            "host1:port",
            "host2:port",
            "host3:port"
        ]
    },
    "task": {
        "index": 1,
        "type": "worker"
    }
}
"""
