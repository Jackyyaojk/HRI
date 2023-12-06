import pybullet as pb
import pybullet_data
import numpy as np
import os

from utils import *



class SimulationEnv():

    def __init__(self, GUI=True):
        
        if GUI is False:
            pb.connect(pb.DIRECT)
        else:
            self.urdfRootPath = pybullet_data.getDataPath()
            pb.connect(pb.GUI, options='--background_color_red=.5 --background_color_green=.5 --background_color_blue=.5')
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI,0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS,0)
            pb.setGravity(0, 0, -9.81)
            self._set_camera()

            # load objects
            self.table = pb.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.62])
            self.panda = PandaSimulator()
            self.visualizeChair()
            self.all_waypoints = {
                                "leg1": [],
                                "leg2": [],
                                "leg3": [],
                                "leg4": [],
                                }
            self.visible_waypoints_idx = [0, total_waypoints-1]
        

    def visualizeWaypoints(self, poses, colors, leg_idx):
        radius = 0.03
        waypoints = []
        for i in range(total_waypoints):
            if i in self.visible_waypoints_idx:
                Id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
                position, color = poses[i], colors[i]
                orientation = pb.getQuaternionFromEuler([0, 0, 0])
                w = pb.createMultiBody(0, Id, -1, position, orientation)
                constraint = pb.createConstraint(w, -1, -1, -1, pb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position)
                pb.changeConstraint(constraint, w, maxForce=0)
                pb.changeVisualShape(w, -1, rgbaColor=color)
            else:
                w = None
            waypoints.append(w)
        for w in waypoints:
            if w:
                for link in range(pb.getNumJoints(self.panda.pandaId)):
                    pb.setCollisionFilterPair(w, self.panda.pandaId, -1, link, 0)
        leg_name = "leg" + str(leg_idx)
        self.all_waypoints[leg_name] = waypoints


    def updateWaypoints(self, new_poses, new_colors, leg_idx):
        leg_name = "leg" + str(leg_idx)
        waypoints = self.all_waypoints[leg_name]
        for i, w in enumerate(waypoints):
            if i in self.visible_waypoints_idx:
                position, color = new_poses[i], new_colors[i]
                orientation = pb.getQuaternionFromEuler([0, 0, 0])
                pb.resetBasePositionAndOrientation(w, position, orientation)
                pb.changeVisualShape(w, -1, rgbaColor=color)
 

 
    def visualizeChair(self):
        self.parts = []
        num_parts = len(LEGS) + 1 
        Panda = TrajectoryClient()
        for i in range(num_parts):
            if i != num_parts - 1:
                ## legs
                position, _, _ = Panda.joint2pose(LEGS[i])
                angle, radius, height = np.pi/2, 0.02, 0.24
                position[-1] = 0.03
            else:
                ## seat
                position = SEAT_POS
                angle, radius, height = 0., 0.13, 0.03
            Id = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=radius, height=height)
            orientation = pb.getQuaternionFromEuler([angle, 0, 0])
            p = pb.createMultiBody(0, Id, -1, position, orientation)
            constraint = pb.createConstraint(p, -1, -1, -1, pb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], position)
            pb.changeConstraint(constraint, p, maxForce=0)
            pb.changeVisualShape(p, -1, rgbaColor=[0.5, 0.25, 0., 1.])
            self.parts.append(p)
        for p in self.parts:
            for link in range(pb.getNumJoints(self.panda.pandaId)):
                pb.setCollisionFilterPair(p, self.panda.pandaId, -1, link, 0)


    def removeWaypoints(self, waypoints):
        for w in waypoints:
            pb.removeBody(w)

    def reset(self):
        self.panda.reset()
        return self.panda.state

    def close(self):
        pb.disconnect()

    def step(self, action):
        joint_pos, grasp_open = action[0], action[1]
        self.panda._position_control(joint_pos, grasp_open)
        pb.stepSimulation()
        next_state = self.panda.state
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        pb.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=165., cameraPitch=-35,
                                     cameraTargetPosition=[0.5, 0, 0.1])
        self.view_matrix = pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = pb.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)
