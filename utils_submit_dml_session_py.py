#!/usr/bin/env python

import json
import requests

import ipfsapi

from tests.testing_utils import serialize_job, make_initialize_job, make_model_json

serialized_job = serialize_job(make_initialize_job(make_model_json()))
mnist_uuid = 'd16c6e86-d103-4e71-8741-ee1f888d206c'
mnist_uuid_2 = '0fcf9cbb-39df-4ad6-9042-a64c87fecfb3'
new_session_key = {"dataset_uuid": mnist_uuid, "label_column_name": "label"}
new_session_key_2 = {"dataset_uuid": mnist_uuid_2, "label_column_name": "label"}
new_session_event = {
        "optimizer_params": {"num_averages_per_round": 0, "max_rounds": 10},
        "serialized_job": serialized_job,
        "participants": ['0fcf9cbb-39df-4ad6-9042-a64c87fecfb3', 
                        'd16c6e86-d103-4e71-8741-ee1f888d206c']
}

client = ipfsapi.connect('127.0.0.1', 5001)
content_hash = client.add_json(new_session_event)
key_hash = client.add_json(new_session_key)
key_hash_2 = client.add_json(new_session_key_2)
assert key_hash != key_hash_2
headers = {
    "Content-Type": "application/json",
}

data = {
    "KEY": str(key_hash),
    "CONTENT": str(content_hash),
}
r = requests.post(
    'http://127.0.0.1:3000/txs',
    headers=headers,
    data=json.dumps(data),
)
data_2 = {
    "KEY": str(key_hash_2),
    "CONTENT": str(content_hash),
}
assert data != data_2
r_2 = requests.post(
    'http://127.0.0.1:3000/txs',
    headers=headers,
    data=json.dumps(data_2),
)
if r.status_code == requests.codes.ok:
    print("OK")
else:
    print("Error")
