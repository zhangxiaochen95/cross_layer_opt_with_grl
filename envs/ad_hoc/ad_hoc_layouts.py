from dataclasses import dataclass, field
from abc import abstractmethod
import random
import numpy as np

from envs.common import *


class AdHocLayout:
    """Base class for all layouts of Ad Hoc Networks"""

    @abstractmethod
    def set_entities(self) -> tuple[ndarray, ndarray, ndarray]:
        """Sets positions of nodes and source/destination of data flows."""
        raise NotImplementedError


@dataclass
class Stationary(AdHocLayout):
    """Stationary layout"""

    range_pos: float = 500  # Range of position (m)
    dim_pos: int = 2  # Dimension of position
    min_link_dist: float = 1  # Minimum distance between nodes (m)
    r_sns: float = 250  # Range of neighbor detection

    n_agt_flows: int = 1  # Number of data flows controlled by agents
    n_amb_flows: int = 1  # Number of ambient data flows
    max_nbrs: int = 10  # Maximum number of neighbors to consider
    n_nodes_per_rgn: tuple = (6, 5, 6, 5, 6, 5, 6, 5, 6)  # Nodes density profiled over subregions
    max_hops: int = 20  # Maximum number of hops

    chan_name: str = 'itu1411'  # Channel model
    p_max_dbm: float = 20  # Maximum Tx power of data flows (dBm)
    n_pow_lvs: int = 4  # Number of discrete power levels for agent flows
    p_bdg: float = float('inf')  # Power budget (Watt)
    n0_dbm: float = -150  # PSD of Rx noise (dBm/Hz)
    p_amb_dbm: float = 10  # Tx power of ambient flows (dBm)

    tot_bw: float = 5e6  # Total available bandwidth (Hz)
    n_chans: int = 1  # Number of sub-channels

    def __post_init__(self):
        self.n_nodes = sum(self.n_nodes_per_rgn)  # Number of nodes
        self.n_flows = self.n_agt_flows + self.n_amb_flows

        # Divide area into subregions.
        self.n_rgns = len(self.n_nodes_per_rgn)  # Number of subregions
        n_divs = int(np.sqrt(len(self.n_nodes_per_rgn)))  # Division in each dimension
        self.range_rgn = self.range_pos / n_divs  # Range of each subregion (m)

        self.p_max = 1e-3 * np.power(10, self.p_max_dbm / 10)  # Tx power of data flows (Watt)
        self.n0 = 1e-3 * np.power(10, self.n0_dbm / 10)  # PSD of Rx noise (Watt/Hz)
        self.p_amb = 1e-3 * np.power(10, self.p_amb_dbm / 10)  # Tx power of ambient flows (Watt)

    def set_stationary_node_positions(self):
        """Evenly sets positions of nodes in each subregion."""
        pos_nodes = []
        # Randomly draw positions of nodes in each subregion.
        for rgn_idx in range(self.n_rgns):
            o_rgn = self.range_rgn * np.array([rgn_idx // 3, rgn_idx % 3])  # Origin of subrigion
            coor_nodes_in_rgn = select_from_box(self.n_nodes_per_rgn[rgn_idx], 0, int(self.range_rgn // self.min_link_dist), self.dim_pos)
            pos_nodes_in_rgn = o_rgn + self.min_link_dist * coor_nodes_in_rgn
            pos_nodes.append(pos_nodes_in_rgn)
        pos_nodes = np.concatenate(pos_nodes)
        return pos_nodes

    def sample_nodes_from_region(self, rgn_idx, n_nodes):
        """Samples distinct nodes in a subregion."""
        nids = random.sample(range(self.n_nodes_per_rgn[rgn_idx]), n_nodes)
        nids = np.array(nids, dtype=int) + sum(self.n_nodes_per_rgn[:rgn_idx])
        return nids

    def set_entities(self):
        pos_nodes = self.set_stationary_node_positions()

        # Ambient flows
        amb_rgn_pairs = random.sample([(2, 4), (6, 4)], 1)
        src_rgn_idx, dst_rgn_idx = amb_rgn_pairs[0]
        amb_src_nids = self.sample_nodes_from_region(src_rgn_idx, self.n_amb_flows)
        amb_dst_nids = self.sample_nodes_from_region(dst_rgn_idx, self.n_amb_flows)

        # Agent flows
        src_rgn_idx, dst_rgn_idx = (0, 8)
        agt_src_nids = self.sample_nodes_from_region(src_rgn_idx, self.n_agt_flows)
        agt_dst_nids = self.sample_nodes_from_region(dst_rgn_idx, self.n_agt_flows)

        src_nids = np.append(amb_src_nids, agt_src_nids)
        dst_nids = np.append(amb_dst_nids, agt_dst_nids)

        return pos_nodes, src_nids, dst_nids


class QuasiStationary(Stationary):
    def set_entities(self):
        pos_nodes = self.set_stationary_node_positions()

        agt_rgn_pairs = random.sample([(0, 8), (8, 0), (2, 6), (6, 2)], 1)[0]
        if agt_rgn_pairs in {(0, 8), (8, 0)}:
            amb_rgn_pairs = random.sample([(2, 4), (6, 4)], 1)[0]
        else:
            amb_rgn_pairs = random.sample([(0, 4), (8, 4)], 1)[0]

        # Ambient flows
        src_rgn_idx, dst_rgn_idx = amb_rgn_pairs
        amb_src_nids = self.sample_nodes_from_region(src_rgn_idx, self.n_amb_flows)
        amb_dst_nids = self.sample_nodes_from_region(dst_rgn_idx, self.n_amb_flows)

        # Agent flows
        src_rgn_idx, dst_rgn_idx = agt_rgn_pairs
        agt_src_nids = self.sample_nodes_from_region(src_rgn_idx, self.n_agt_flows)
        agt_dst_nids = self.sample_nodes_from_region(dst_rgn_idx, self.n_agt_flows)

        src_nids = np.append(amb_src_nids, agt_src_nids)
        dst_nids = np.append(amb_dst_nids, agt_dst_nids)

        return pos_nodes, src_nids, dst_nids


@dataclass
class Clusters(AdHocLayout):
    range_pos: float = 500  # Range of position (m)
    dim_pos: int = 2  # Dimension of position
    min_link_dist: float = 1  # Minimum distance between nodes (m)
    r_sns: float = 250  # Range of neighbor detection

    n_agt_flows: int = 1  # Number of data flows controlled by agents
    n_amb_flows: int = 0  # Number of ambient data flows
    max_nbrs: int = 10  # Maximum number of neighbors to consider
    n_nodes_per_cl: tuple = (5, 3, 3, 4, 4, 5)
    max_hops: int = 20  # Maximum number of hops

    chan_name: str = 'itu1411'  # Channel model
    p_max_dbm: float = 20  # Maximum Tx power of data flows (dBm)
    n_pow_lvs: int = 4  # Number of discrete power levels for agent flows
    p_bdg: float = float('inf')  # Power budget (Watt)
    n0_dbm: float = -150  # PSD of Rx noise (dBm/Hz)
    p_amb_dbm: float = 10  # Tx power of ambient flows (dBm)

    tot_bw: float = 5e6  # Total available bandwidth (Hz)
    n_chans: int = 1  # Number of sub-channels

    def __post_init__(self):

        self.n_cls = len(self.n_nodes_per_cl)
        self.n_nodes = sum(self.n_nodes_per_cl)  # Number of nodes
        self.n_flows = self.n_agt_flows + self.n_amb_flows

        self.p_max = 1e-3 * np.power(10, self.p_max_dbm / 10)  # Tx power of data flows (Watt)
        self.n0 = 1e-3 * np.power(10, self.n0_dbm / 10)  # PSD of Rx noise (Watt/Hz)
        self.p_amb = 1e-3 * np.power(10, self.p_amb_dbm / 10)  # Tx power of ambient flows (Watt)

    def set_entities(self) -> tuple[ndarray, ndarray, ndarray]:
        pos_cls = []
        # # range_pos = 600m
        # pos_cls.append(np.array([[300, 50], [100, 100], [500, 100]]))
        # sign = 2 * (np.random.randint(0, 2) - 0.5)
        # pos_cls.append(np.array([[300 + sign * 200, 300], [300 + sign * 200, 500]]))
        # pos_cls.append(np.array([[300, 550]]))

        # range_pos = 500m
        pos_cls.append(np.array([[250, 50], [50, 100], [450, 100]]))
        sign = 2 * (np.random.randint(0, 2) - 0.5)
        pos_cls.append(np.array([[250 + sign * 200, 250], [250 + sign * 200, 450]]))
        pos_cls.append(np.array([[250, 450]]))

        pos_cls = np.concatenate(pos_cls, axis=0)

        pos_nodes = []
        for cl_idx, pos_cl in enumerate(pos_cls):
            r_cls = 100 if cl_idx == 0 else 75
            pos_nodes.append(pos_cl + select_from_box(self.n_nodes_per_cl[cl_idx], 0, r_cls, 2) - r_cls / 2)
        pos_nodes = np.concatenate(pos_nodes, axis=0)

        x, y = pos_nodes[:, 0], pos_nodes[:, 1]
        ang = np.random.randint(0, 4) * np.pi / 2
        x_o, y_o = self.range_pos / 2, self.range_pos / 2
        x_rot = (x - x_o) * np.cos(ang) + (y - y_o) * np.sin(ang) + x_o
        y_rot = - (x - x_o) * np.sin(ang) + (y - y_o) * np.cos(ang) + y_o
        pos_nodes = np.vstack((x_rot, y_rot)).T
        pos_nodes = np.clip(pos_nodes, 0, self.range_pos)

        src_nids = np.array([0])
        dst_nids = np.array([self.n_nodes - 1])
        return pos_nodes, src_nids, dst_nids


kwargs_1flow = dict(n_amb_flows=0, n_nodes_per_rgn=tuple([3] * 9), max_nbrs=6)
SCENARIOS = {
    'debug': Stationary(max_nbrs=4, n_chans=3, n_pow_lvs=2, n_nodes_per_rgn=tuple([3] * 9)),
    'debug-full-pow': Stationary(max_nbrs=4, n_chans=3, n_pow_lvs=1, n_nodes_per_rgn=tuple([3] * 9)),

    '1flow': QuasiStationary(n_amb_flows=0, n_nodes_per_rgn=tuple([3] * 9), max_nbrs=6,),
    '1flow-full-pow': QuasiStationary(n_amb_flows=0, n_nodes_per_rgn=tuple([3] * 9), max_nbrs=6, n_pow_lvs=1),

    'cls': Clusters(),
    'cls-full-pow': Clusters(n_pow_lvs=1),

    '2flows': QuasiStationary(),
    '2flows-full-pow': QuasiStationary(n_pow_lvs=1),

    '1f': QuasiStationary(n_amb_flows=0, n_nodes_per_rgn=tuple([3] * 9), max_nbrs=6,),
    '1f-full-pow': QuasiStationary(n_amb_flows=0, n_nodes_per_rgn=tuple([3] * 9), max_nbrs=6, n_pow_lvs=1),

    '1fb': QuasiStationary(n_amb_flows=0),
    '1fb-full-pow': QuasiStationary(n_amb_flows=0, n_pow_lvs=1),
}
