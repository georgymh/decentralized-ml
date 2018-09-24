import asyncio
from kademlia.network import Server


CONFIG = json.load(p2p_config)

class Node(object):
    def __init__(self):
        self.node = Server()
        self.loop = asyncio.get_event_loop()
        try:
            self.node.listen()
            self.loop.run_until_complete(self.node.bootstrap([(CONFIG.host, CONFIG.port)]))
        except:
            print("Peer-to-Peer sockets not started.")

    def set_val(self, key, value) -> None:
        self.loop.run_until_complete(self.node.set(key, value))

    def get_val(self, key):
        retval = self.loop.run_until_complete(self.node.get(key))
        return retval
