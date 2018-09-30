import tests.context
import pytest

from core.blockchain.ipfs_utils import IPFS


def test_sanity():
    ipfs_client = IPFS()
    obj_to_store = 'foo'
    address = ipfs_client.upload(obj_to_store)
    retrieved_obj = ipfs_client.download(address)
    assert retrieved_obj == obj_to_store
