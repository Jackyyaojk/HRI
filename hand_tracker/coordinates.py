import numpy as np



class Coordinates(object):

    def __init__(self):
        self.transforms = {}

    def init_global_tracker(self, g0_r, gxa_r):
        x_delta_r = gxa_r[0] - g0_r[0]
        y_delta_r = gxa_r[1] - g0_r[1]
        slope = y_delta_r / x_delta_r
        phi = np.arctan(slope)

        if y_delta_r > 0 and x_delta_r > 0:
            theta = -phi
        elif y_delta_r > 0 and x_delta_r < 0:
            theta = -np.pi - phi
        elif y_delta_r < 0 and x_delta_r < 0:
            theta = np.pi - phi
        elif y_delta_r < 0 and x_delta_r > 0:
            theta = -phi

        cos_theta = np.cos(theta)[0]
        sin_theta = np.sin(theta)[0]
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0.],
                                    [-sin_theta, -cos_theta, 0.],
                                    [0., 0., 1.]])
        translation_matrix = -np.matmul(rotation_matrix, g0_r)
        wide_matrix = np.hstack([rotation_matrix, translation_matrix])
        self.transforms["tracker"] = np.vstack([wide_matrix, [0.0, 0.0, 0.0, 1.0]])

    def global_pos_tracker(self, p_r):
        p_r = np.vstack((p_r, np.array([1])))
        p_g = np.matmul(self.transforms["tracker"], p_r)
        return np.delete(p_g, 3, 0)
