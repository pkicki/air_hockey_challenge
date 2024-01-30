import os
import sys
from time import perf_counter
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

from constants import Base
import matplotlib.pyplot as plt

import hpo_opt_new as hpoo

urdf_path = os.path.join(os.path.dirname(__file__), "iiwa_striker.urdf")


def get_hitting_configuration_opt(x, y, z, th, q0=None):
    if q0 is None:
        q0 = Base.configuration
    q0 = q0 + [0.] * (9 - len(q0))
    s = hpoo.optimize(urdf_path, x, y, z, np.cos(th), np.sin(th), q0)
    if not s:
        return None, None
    q = s[:7]
    q_dot = np.array(s[9:16])
    return q, q_dot.tolist()

if __name__ == "__main__":
    x = 1.01
    y = 0.
    th = 0.
    q, q_dot = get_hitting_configuration_opt(x, y, 0.16, th)
    print(q)
    print(q_dot)