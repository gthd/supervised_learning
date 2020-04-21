import numpy as np
import vrep
import buffer

class Script:

    def __init__(self, my_robot):
        self.robot = my_robot
        self.buffer = buffer.ReplayMemory(100)
        self.client_id = self.robot.client_id
        self.states = []
        self.object_position = None
        self.euler_angles2 = None
        self.pickup_position = None
        self.pickup_orientation = None
        self.image = None

    global COUNTER
    COUNTER = 0
    global COUNTER1
    COUNTER1 = 0
    def success_rate(self, label):
        global COUNTER
        global COUNTER1
        COUNTER += 1
        if label:
            COUNTER1 += 1
        if COUNTER % 100 == 0:
            print('success ratio is: ', (float(COUNTER1)/float(COUNTER)))

    def get_sawyer_position(self):
        _, sawyer_target_position = vrep.simxGetObjectPosition(self.robot.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        return sawyer_target_position

    def get_sawyer_orientation(self):
        _, sawyer_target_orientation = vrep.simxGetObjectOrientation(self.robot.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        return sawyer_target_orientation

    def get_object_position(self):
        _, object_position = vrep.simxGetObjectPosition(self.robot.client_id, \
            self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
        return object_position

    def get_object_orientation(self):
        _, object_orientation = vrep.simxGetObjectOrientation(self.robot.client_id, \
            self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
        return object_orientation

    def new_episode(self, label):
        self.buffer.push(label, self.pickup_position[0], self.pickup_position[1], \
            self.pickup_orientation[1], self.image)
        self.buffer.store_at_disk()
        self.success_rate(label)

    def pick_position(self, exploit=True):

        _, self.object_position = vrep.simxGetObjectPosition(self.client_id, \
            self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
        _, sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        if exploit == bool(1):
            self.pickup_position = np.array([self.object_position[0], self.object_position[1], \
                sawyer_target_position[2]])
        else:
            # random = np.random.normal(0,0.05,3)
            # self.pickup_position = np.array([self.object_position[0] + random[0]
            # self.object_position[1] + random[1], sawyer_target_position[2]])

            action_x = np.random.uniform(low=1.028, high=1.242, size=1)[0]
            action_y = np.random.uniform(low=1.1, high=1.278, size=1)[0]
            self.pickup_position = np.array([action_x, action_y, sawyer_target_position[2]])

    def pick_orientation(self, exploit=True):
        _, euler_angles = vrep.simxGetObjectOrientation(self.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        _, self.euler_angles2 = vrep.simxGetObjectOrientation(self.client_id, \
            self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
        if exploit == bool(1):
            self.pickup_orientation = np.array([euler_angles[0], self.euler_angles2[1], \
                euler_angles[2]])
        else:
            ori = np.random.uniform(0.017, 1.553, 1)[0]
            self.pickup_orientation = np.array([euler_angles[0], ori, euler_angles[2]])

    def set_gripper_position(self):
        _, sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        self.image = self.robot.get_image()
        move_direction = np.asarray([self.pickup_position[0] - sawyer_target_position[0], \
            self.pickup_position[1] - sawyer_target_position[1], self.pickup_position[2] - \
            sawyer_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.03*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.03))
        remaining_magnitude = -num_move_steps * 0.03 + move_magnitude
        remaining_distance = remaining_magnitude * move_direction/move_magnitude

        for step_iter in range(num_move_steps): #selects action and executes action
            vrep.simxSetObjectPosition(self.client_id, self.robot.sawyer_target_handle, -1, \
                (sawyer_target_position[0] + move_step[0], sawyer_target_position[1] + \
                move_step[1], sawyer_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
            _, sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, \
                self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

        vrep.simxSetObjectPosition(self.robot.client_id, self.robot.sawyer_target_handle, -1, \
        (sawyer_target_position[0] + remaining_distance[0], sawyer_target_position[1] + \
        remaining_distance[1], sawyer_target_position[2]+ remaining_distance[2]), \
        vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def set_gripper_orientation(self):
        _, sawyer_orientation = vrep.simxGetObjectOrientation(self.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (self.pickup_orientation[1] - sawyer_orientation[1] > 0) \
            else -0.3
        num_rotation_steps = int(np.floor((self.pickup_orientation[1] - \
            sawyer_orientation[1])/rotation_step))

        for step_iter in range(num_rotation_steps):
            vrep.simxSetObjectOrientation(self.robot.client_id, \
                self.robot.sawyer_target_handle, -1, (sawyer_orientation[0], \
                sawyer_orientation[1] + rotation_step, sawyer_orientation[2]), \
                vrep.simx_opmode_blocking)
            _, sawyer_orientation = vrep.simxGetObjectOrientation(self.client_id, \
                self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

        vrep.simxSetObjectOrientation(self.robot.client_id, self.robot.sawyer_target_handle, \
            -1, (sawyer_orientation[0], self.pickup_orientation[1], sawyer_orientation[2]), \
            vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def move_down(self): #3 time-steps
        _, object_position = vrep.simxGetObjectPosition(self.client_id, \
            self.robot.object_handle[0], -1, vrep.simx_opmode_blocking)
        _, sawyer_target_position = vrep.simxGetObjectPosition(self.client_id, \
            self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)

        move_direction = np.asarray([self.pickup_position[0] - sawyer_target_position[0], \
            self.pickup_position[1] - sawyer_target_position[1], object_position[2] + 0.01 \
            - sawyer_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.03*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.03))
        remaining_magnitude = -num_move_steps * 0.03 + move_magnitude
        remaining_distance = remaining_magnitude * move_direction/move_magnitude

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.client_id, self.robot.sawyer_target_handle,\
                -1, (sawyer_target_position[0] + move_step[0], sawyer_target_position[1] \
                + move_step[1], sawyer_target_position[2] + move_step[2]), \
                vrep.simx_opmode_blocking)
            _, sawyer_target_position = vrep.simxGetObjectPosition(self.client_id,\
                self.robot.sawyer_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSynchronousTrigger(self.client_id)
            vrep.simxGetPingTime(self.client_id)

        vrep.simxSetObjectPosition(self.robot.client_id, self.robot.sawyer_target_handle, \
            -1, (sawyer_target_position[0] + remaining_distance[0], sawyer_target_position[1] \
            + remaining_distance[1], sawyer_target_position[2]+ remaining_distance[2]), \
            vrep.simx_opmode_blocking)
        vrep.simxSynchronousTrigger(self.client_id)
        vrep.simxGetPingTime(self.client_id)

    def open_hand(self):
        self.robot.open_hand()

    def close_hand(self):
        self.robot.close_hand()

    def lift_arm(self):
        self.robot.lift_arm()

    def successful_grasp(self):
        label = self.robot.successful_grasp()
        return label
