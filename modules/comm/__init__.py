REGISTRY = {}

from modules.comm.tarmac import TarMAC
REGISTRY['tarmac'] = TarMAC

from modules.comm.disc import DiscreteCommunication
REGISTRY['disc'] = DiscreteCommunication
