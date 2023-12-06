import numpy as np

import hand_tracker.coordinates as coordinates
import hand_tracker.triad_openvr as triad_openvr


class TrackerState(object):

    def __init__(self):
        self.Coords = coordinates.Coordinates()
        self.track = triad_openvr.triad_openvr()

    def set_global_coords(self, g0_r, gxa_r):
        self.Coords.init_global_tracker(g0_r, gxa_r)

    def auto_set_global_coords(self):
        input("Place the tracker at the global origin then press enter.")
        pose_r = self.track.devices["tracker_1"].get_pose_euler()
        g0_r = np.array([[pose_r[0]], [pose_r[2]], [pose_r[1]]])
        print("g0_r: ", g0_r)
        input("Place the tracker at some nonzero point along the x-axis.")
        pose_r = self.track.devices["tracker_1"].get_pose_euler()
        gxa_r = np.array([[pose_r[0]], [pose_r[2]], [pose_r[1]]])
        print("gxa_r: ", gxa_r)
        self.Coords.init_global_tracker(g0_r, gxa_r)
        return g0_r, gxa_r

    def get_tracker_pos(self):
        pose_r = self.track.devices["tracker_1"].get_pose_euler()
        try:
            tracker_pos_r = np.array([[pose_r[0]], [pose_r[2]], [pose_r[1]]])
            tracker_pos_g = self.Coords.global_pos_tracker(tracker_pos_r)
            return tracker_pos_g, True
        except:
            return np.array([[0], [0], [0]]), False

    def show_running_pos(self):
        tracker_pos_g = self.get_tracker_pos()
        txt = "Global Values: "
        txt += "%.4f" % tracker_pos_g[0] + " "
        txt += "%.4f" % tracker_pos_g[1] + " "
        txt += "%.4f" % tracker_pos_g[2] + " "

        print("\r" + txt, end="")

    def return_pos_txt(self):
        tracker_pos_g = self.get_tracker_pos()
        txt = "Global Values: "
        txt += "%.4f" % tracker_pos_g[0] + " "
        txt += "%.4f" % tracker_pos_g[1] + " "
        txt += "%.4f" % tracker_pos_g[2] + " "
        return txt
