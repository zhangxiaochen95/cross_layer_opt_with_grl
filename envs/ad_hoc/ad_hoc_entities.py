from typing import Optional
import numpy as np
from numpy import ndarray


class Node:
    """Wireless node in Ad Hoc networks"""

    legal_roles = {'src', 'rly', 'dst'}  # Legal roles of nodes are source/relay/destination

    def __init__(self, nid: int, n_chans: int):

        self.nid = nid  # Identifier of node
        self.n_chans = n_chans  # Number of sub-channels
        self.pos: Optional[ndarray] = None
        self.role: str = 'rly'

        self.p_tx = np.zeros(self.n_chans, dtype=np.float32)  # Tx power on each sub-channel (Watt)
        self.idle = np.zeros(self.n_chans, dtype=bool)

    def reset(self, pos_node: ndarray):
        self.pos = pos_node
        self.role = 'rly'
        self.p_tx = np.zeros(self.n_chans, dtype=np.float32)
        self.idle = np.ones(self.n_chans, dtype=bool)


class Link:
    """Wireless link between a transceiver pair"""

    def __init__(self, tx_node: Node, rx_node: Node, chan_idx: int, p_tx: float):
        self.tx: Node = tx_node
        self.rx: Node = rx_node
        self.chan_idx: int = chan_idx

        self.tx.p_tx[self.chan_idx] = p_tx
        self.tx.idle[self.chan_idx] = False
        self.rx.idle[self.chan_idx] = False

    @property
    def p_tx(self) -> float:
        return self.tx.p_tx[self.chan_idx]

    def remove(self):
        self.tx.p_tx[self.chan_idx] = 0
        self.tx.idle[self.chan_idx] = True
        self.rx.idle[self.chan_idx] = True


class Flow:
    """Data flow with a multi-hop route"""

    def __init__(self,
                 fid: int,  # Identifier of node
                 n_chans: int,  # Number of sub-channels
                 n_nodes: int,  # Number of nodes
                 p_levels: list[float],  # Discrete levels of transmit power (Watt)
                 p_budget: float,  # Total power budget of the flow (Watt)
                 allow_full_duplex: bool = False,  # Whether to allow simultaneous Tx/Rx on one channel
                 ):

        self.fid = fid
        self.n_chans = n_chans
        self.n_nodes = n_nodes

        self.route: list[Link] = []  # Route list
        self.src: Optional[Node] = None  # Source node
        self.dst: Optional[Node] = None  # Destination node

        self.qual_nodes: ndarray = np.ones(self.n_nodes, dtype=bool)  # Qualification of candidate nodes
        self.p_bdg = p_budget
        self.p_lvs = p_levels

        self._allow_full_duplex = allow_full_duplex

    def reset(self, src_node: Node, dst_node: Node):
        # Assign source/destination.
        src_node.role = 'src'
        self.src = src_node
        dst_node.role = 'dst'
        self.dst = dst_node

        # Clear the route.
        self.clear_route()

    def clear_route(self):
        """Clears route."""
        # Remove all links in the route.
        for link in self.route:
            link.remove()
        self.route = []  # Empty route.

        # All nodes but source can be added to route in the future.
        self.qual_nodes = np.ones(self.n_nodes, dtype=bool)
        self.qual_nodes[self.src.nid] = 0

    def check(self, node: Node) -> bool:
        """Checks whether a node is qualified to be added."""
        if node.nid in self.nids_in_route:
            return False
        if (node.role == 'rly') and self.qual_nodes[node.nid]:
            if self._allow_full_duplex:  # Simultaneous transmitting/receiving on the same channel is allowed.
                if node.idle.any():
                    return True
            else:  # Transmitting/receiving must take different sub-channels.
                # Then, a qualified next relay must meet two criteria:
                # 1) An idle sub-channel is shared by next/front nodes;
                # 2) Another sub-channel is available for later hop.
                if (node.idle.sum() >= 2) and (node.idle * self.front.idle).any():
                    return True
        elif node is self.dst:
            if self._allow_full_duplex:
                if node.idle.any():
                    return True
            else:
                if (node.idle * self.front.idle).any():
                    return True
        else:
            return False

    def ban(self, node: Node) -> None:
        """Bans a node from the route."""
        self.qual_nodes[node.nid] = False

    def add_hop(self, next_node: Node, chan_idx: int, p_idx: int) -> None:
        """Adds a hop to the route."""
        # Create a link between current front node and next node.
        link = Link(self.front, next_node, chan_idx, self.p_lvs[p_idx])
        # Add link to route.
        self.route.append(link)
        # Disqualify added node to prevent loop in the route.
        self.qual_nodes[next_node.nid] = 0

    @property
    def n_hops(self) -> int:
        """Number of hops"""
        return len(self.route)

    @property
    def is_connected(self) -> bool:
        """Returns whether source node is connected to destination with current route"""
        return self.front == self.dst

    @property
    def front(self) -> Node:
        """Current frontier node"""
        return self.src if len(self.route) == 0 else self.route[-1].rx

    @property
    def n_pow_lvs(self) -> int:
        """Number of Tx power levels"""
        return len(self.p_lvs)

    @property
    def p_tot(self) -> float:
        """Total power cost"""
        return sum([link.p_tx for link in self.route])

    @property
    def p_rem(self) -> float:
        """Remaining power budget"""
        return self.p_bdg - self.p_tot

    @property
    def nids_in_route(self):
        return [self.src.nid] + [link.rx.nid for link in self.route]

    def __repr__(self):
        return f"Flow-{self.fid} from n{self.src.nid} to n{self.dst.nid} with route {self.nids_in_route}"
