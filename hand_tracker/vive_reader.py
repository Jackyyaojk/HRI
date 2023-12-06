
import numpy as np

from tracker_state import TrackerState
from utils import *



tracker = TrackerState()
tracker.set_global_coords(GLOBAL_O, GLOBAL_X)


# global_O, global_X = tracker.auto_set_global_coords()


while True:

    # Read Vive
    vive_pos_raw, valid = tracker.get_tracker_pos()
    if valid:
        position_hand = vive_pos_raw.squeeze()
        print(position_hand)
    else:
        print('No data from Vive')
        continue
