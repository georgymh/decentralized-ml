import base58
import ipfsapi
import json
import os
from keras.models import load_model, model_from_json


with open("./core/blockchain/ipfs_config.json", "r") as read_file:
    CONFIG = json.load(read_file)

class IPFS(object):
    # OBJ_TYPES = {str: IPFS._cast_str, dict: IPFS._cast_dict}

    def __init__(self):
        '''TODO: Decide if `kv` is needed
        TODO: Refactor dependencies
        '''
        self.kv = {}
        self.client = None
        try:
            self.client = ipfsapi.connect(CONFIG['host'], CONFIG['port'])
        except Exception as e:
            print("IPFS daemon not started.")

    def upload_local(self, key: str, obj: object):
        '''Test method, DO NOT USE
        '''
        addr = self.client.add(obj)
        self.kv[key] = addr

    def download_local(self, key: str) -> object:
        '''Test method, DO NOT USE
        '''
        addr = self.kv[key]
        return self.client.get(addr)

    def upload(self, obj: object) -> str:
        '''Provided any Python object, store it on IPFS and then upload
        the hash to the blockchain
        TODO: blockchain setter API required
        '''
        ipfs_hash = self._content_to_ipfs(obj)
        byte_addr = self._ipfs_to_blockchain(ipfs_hash)
        return addr

    def download(self, byte_addr: bytes) -> object:
        '''Provided an on-chain content address, retrieve the Python
        object from IPFS
        TODO: blockchain getter API required
        '''
        ipfs_hash = self._blockchain_to_ipfs(byte_addr)
        content = self._ipfs_to_content(ipfs_hash)
        return content

    def _ipfs_to_content(self, ipfs_hash: str) -> object:
        '''Helper function to retrieve a Python object from an IPFS hash
        '''
        return self.client.get(addr)

    def _content_to_ipfs(self, obj: object) -> str:
        '''Helper function to deploy a Python object onto IPFS, 
        returns an IPFS hash
        '''
        return self.client.add(obj)

    def _ipfs_to_blockchain(self, ipfs_hash: str) -> bytes:
        '''Helper function to convert IPFS hashes to on-chain content addresses
        '''
        return base58.b58encode(b'\x12 ' + ipfs_hash)

    def _blockchain_to_ipfs(self, byte_addr: bytes) -> str:
        '''Helper function to convert on-chain content addresses to IPFS hashes
        '''
        return base58.b58decode(byte_addr)[2:]

    def _cast_str(self, input: str):
        '''Helper function to cast string inputs to desired type
        '''
        pass

    def _cast_dict(self, input: dict):
        '''Helper function to cast dict inputs to desired type
        '''
        pass
