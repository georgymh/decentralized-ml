import json
import logging
import requests
import time
from typing import Callable
from enum import Enum


logging.basicConfig(level=logging.DEBUG,
    format='[BlockchainUtils] %(message)s')


##############################################################################
###                              REQUIRE IPFS                              ###
##############################################################################

def upload(client: object, value: dict) -> str:
    """
    Provided any Python object, store it on IPFS and then upload the hash that
    will be uploaded to the blockchain as a value
    """
    # assert TxEnum.KEY.name in value
    # assert TxEnum.CONTENT.name in value
    ipfs_hash = content_to_ipfs(client, value)
    return str(ipfs_hash)

def download(client: object, key: str, state: list) -> list:
    """
    Provided an on-chain key, retrieve the value from local state and retrieve
    the Python object from IPFS
    TODO: implement a better way to parse through state list
    """
    relevant_txs = list(
        map(lambda tx: ipfs_to_content(client, tx.get(TxEnum.CONTENT.name)),
            filter(lambda tx: tx.get(TxEnum.KEY.name) == key, state)))
    return relevant_txs

def ipfs_to_content(client: object, ipfs_hash: str) -> object:
    """
    Helper function to retrieve a Python object from an IPFS hash
    """
    return client.get_json(ipfs_hash)

def content_to_ipfs(client: object, content: dict) -> str:
    """
    Helper function to deploy a Python object onto IPFS, returns an IPFS hash
    # TODO: Figure out how to calculate the multihash without actually pushing
    the object to IPFS. This is possible with 'ipfs add -n' via CLI.
    """
    return client.add_json(content)

##############################################################################
###                                 DIFFS                                  ###
##############################################################################

def get_diffs(global_state: list, local_state: list) -> list:
    """
    Return list of transactions that are present in `global_state` but not in
    `local_state`
    """
    return list(filter(lambda elem: elem not in local_state, global_state))

# TODO: consider merging the two methods below into one

def filter_diffs(global_state_wrapper: object, local_state: list,
                    filter_method: Callable = lambda tx: True,
                    map_method: Callable = lambda tx: tx) -> list:
    """
    Provided the freshly-downloaded state, call a handler on each transaction
    that was not already present in our own state and return the new state
    """
    state_diffs = get_diffs(global_state_wrapper.get(TxEnum.MESSAGES.name, {}), 
                            local_state)
    return (state_diffs, list(map(map_method, filter(filter_method, state_diffs))))

##############################################################################
###                                REQUESTS                                ###
##############################################################################

def construct_getter_call(host: str, port: int) -> str:
    return "http://{0}:{1}/state".format(host, port)

def make_getter_call(host: str, port: int) -> object:
    tx_receipt = requests.get(construct_getter_call(host, port))
    tx_receipt.raise_for_status()
    return tx_receipt

def construct_setter_call(host: str, port: int) -> str:
    return "http://{0}:{1}/txs".format(host, port)

def make_setter_call(host: str, port: int, tx: dict) -> object:
    tx_receipt = requests.post(construct_setter_call(host, port), json=tx)
    tx_receipt.raise_for_status()
    return tx_receipt

def get_global_state(host: str, port: int, timeout: int) -> object:
    """
    Gets the global state which should be a list of dictionaries
    TODO: perhaps it might be better to offload the retrying to the `getter`
    """
    timeout = time.time() + timeout
    tx_receipt = None
    while time.time() < timeout:
        try:
            tx_receipt = make_getter_call(host, port).json()
            break
        except (UnboundLocalError, requests.exceptions.ConnectionError) as e:
            logging.info("HTTP GET error, got: {0}".format(e))
            continue
    return tx_receipt

def getter(client: object, key: str, local_state: list, port: int, timeout: int,
            host: str = '127.0.0.1') -> list:
    """
    Provided a key, get the IPFS hash from the blockchain and download the
    object from IPFS. This pulls from the global state but DOES NOT update the
    local state
    """
    state_diffs, filtered_diffs = filter_diffs(get_global_state(host, port, timeout),
                                                local_state)
    new_state = local_state + state_diffs
    return download(client, key, new_state)

def setter(client: object, key: str, port: int, round_num: int, value: object,
            flag: bool = False, host: str = '127.0.0.1', state_append: object = False) -> str:
    """
    Provided a key and a JSON/np.array object, upload the object to IPFS and
    then store the hash as the value on the blockchain. The key should be a
    backward reference to a prior tx unless flag is set, in which case the key
    is also stored onto the blockchain
    TODO: decide on error handling that matches `get_global_state`
    """
    logging.info("Setting to blockchain...")
    on_chain_value = upload(client, value) if value else None
    key = upload(client, key) if flag else key
    tx = Transaction(key, on_chain_value, round_num)
    tx_receipt = None
    try:
        tx_receipt = make_setter_call(host, port, tx.get_tx())
    except Exception as e:
        logging.info("HTTP POST error, got: {0}".format(e))
    if state_append != False:
        state_append.append(tx.get_tx())
    return tx_receipt.text

##############################################################################
###                              OBJECTS                                   ###
##############################################################################

class TxEnum(Enum):
    """
    Enum of tx-related constants, e.g. standard keys present in a transaction
    """
    KEY = "KEY"
    CONTENT = "CONTENT"
    ROUND = "ROUND"
    MESSAGES = "MESSAGES"

class Transaction(object):
    """
    Object that represents transactions with the specification outlined:
        `{
            'key': ...
            'content': ...
         }`
    """
    def __init__(self, key: str, content: str, round_num: int) -> None:
        self.key = key
        self.content = content
        self.round_num = round_num

    def get_tx(self) -> dict:
       return {TxEnum.KEY.name: self.key, TxEnum.CONTENT.name: self.content,
                TxEnum.ROUND.name: self.round_num}
