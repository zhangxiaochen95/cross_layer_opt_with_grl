import numpy as np
from numpy import ndarray


CHANNEL_MODELS = {}


class FSPLChannel(object):
    """Free space path loss (FSPL) channel model"""
    def __init__(self, fc: float = 2.4e9) -> None:
        self.fc = fc  # Central carrier frequency (Hz)

    def estimate_chan_gain(self, d: ndarray) -> ndarray:
        fspl = (4 * np.pi * self.fc * d / 3e8) ** 2  # The Friis equation
        return 1 / fspl


CHANNEL_MODELS['fspl'] = FSPLChannel


class ITU1411Channel(object):
    """Short-range outdoor model ITU-1411"""

    def __init__(self, fc: float = 2.4e9, antenna_gain_db: float = 2.5, antenna_height: float = 1.5):
        self.fc = fc  # Carrier frequency (Hz)
        self.ant_gain_db = antenna_gain_db  # Antenna gain in dBi
        self.h_ant = antenna_height  # Height of antenna (m)

    def estimate_chan_gain(self, d: ndarray) -> ndarray:
        h1, h2 = self.h_ant, self.h_ant
        signal_lambda = 2.998e8 / self.fc
        # compute relevant quantity.
        Rbp = 4 * h1 * h2 / signal_lambda
        Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
        sum_term = 20 * np.log10(d / Rbp)
        Tx_over_Rx = Lbp + 6 + sum_term + ((d > Rbp).astype(int)) * sum_term  # Adjust for longer path loss
        pl = -Tx_over_Rx + self.ant_gain_db  # Add antenna gain
        return np.power(10, (pl / 10))  # convert from decibel to absolut


CHANNEL_MODELS['itu1411'] = ITU1411Channel


class AirToGroundChannel(object):
    """Air-to-ground (ATG) channel model proposed in "Optimal LAP Altitude for Maximum Coverage".
    Parameter a and b are provided in paper
    "Efficient 3-D Placement of an Aerial Base Station in Next Generation Cellular Networks".
    """

    ATG_CHAN_PARAMS = {
        'suburban': {'a': 4.88, 'b': 0.43, 'eta_los': 0.1, 'eta_nlos': 21},
        'urban': {'a': 9.61, 'b': 0.16, 'eta_los': 1, 'eta_nlos': 20},
        'dense-urban': {'a': 12.08, 'b': 0.11, 'eta_los': 1.6, 'eta_nlos': 23},
        'high-rise-urban': {'a': 27.23, 'b': 0.08, 'eta_los': 2.3, 'eta_nlos': 34}
    }

    def __init__(self, scene: str = 'urban', fc: float = 2.4e9) -> None:
        # Set scene-specific parameters.
        for k, v in self.ATG_CHAN_PARAMS[scene].items():
            self.__setattr__(k, v)
        self.fc = fc  # Central carrier frequency (Hz)

    def estimate_chan_gain(self, d_ground: ndarray, h_ubs: float) -> ndarray:
        """Estimates the channel gain from horizontal distance."""
        # Get direct link distance.
        d_link = np.sqrt(np.square(d_ground) + np.square(h_ubs))
        # Estimate probability of LoS link emergence.
        p_los = 1 / (1 + self.a * np.exp(-self.b * (180 / np.pi * np.arcsin(h_ubs / d_link) - self.a)))
        # Compute free space path loss (FSPL).
        fspl = (4 * np.pi * self.fc * d_link / 3e8) ** 2
        # Path loss is the weighted average of LoS and NLoS cases.
        pl = p_los * fspl * 10 ** (self.eta_los / 10) + (1 - p_los) * fspl * 10 ** (self.eta_nlos / 10)
        return 1 / pl


CHANNEL_MODELS['a2g'] = AirToGroundChannel


def compute_antenna_gain(theta: ndarray, psi: float) -> ndarray:
    """Computes antenna gain using simple two-lobe approximation.
    Direction of UBS antenna is perpendicular to ground.
    See "Joint Altitude and Beamwidth Optimization for UAV-Enabled Multiuser Communications" for more explanation.
    Args:
        theta (float): elevation angle from GTs to UBSs between (0, pi/2].
        psi (float): Half-beamwidth of directional antennas (rad)
    """
    g_main = 2.285 / np.power(psi, 2)  # Constant gain of main lobe
    g_side = 0  # Ignored gain of side lobe
    return ((np.pi / 2 - theta) <= psi) * g_main + ((np.pi / 2 - theta) > psi) * g_side


if __name__ == '__main__':
    d = np.arange(0, 1000, 100)
    print(d)
    chan_model = AirToGroundChannel()
    g = chan_model.estimate_chan_gain(d, 100)
    print(10 * np.log10(g))