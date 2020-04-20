import robot
import robot2
import numpy as np

class Policy:

    def __init__(self):
        self.robot = robot.VrepCommunication()
        self.termination_height = 0.13
        self.script = ''

    def start_vrep(self, again=False):
        if again:
            self.robot.end_communication()
            self.robot.establish_communication()
            self.robot.start_vrep()
            self.robot.initialise()
            self.robot.pick_color()
            self.robot.add_object()
            self.robot.reset_object_state()
            self.robot.get_initial_position()
            self.script = robot2.Script(self.robot)
        else:
            self.robot.establish_communication()
            self.robot.start_vrep()
            self.robot.initialise()
            self.robot.pick_color()
            self.robot.add_object()
            self.robot.reset_object_state()
            self.robot.get_initial_position()
            self.script = robot2.Script(self.robot)

    # This method is for exploiting the script that produces transitions
    # that are part of successful grasps.
    global COUNTER1
    COUNTER1 = 0
    def exploit(self):
        global COUNTER1
        print('Exploiting .. no' + str(COUNTER1))
        COUNTER1 += 1
        self.script.pick_position()
        self.script.pick_orientation()
        self.script.set_gripper_position()
        self.script.set_gripper_orientation()
        self.script.move_down()
        self.script.close_hand()
        self.script.lift_arm()
        label = self.script.successful_grasp()
        self.script.new_episode(label)
        self.robot.set_initial_position()

    # This method is for exploring the action-space and for providing the agent with
    # new unseen data for dealing with the confounding error problem.
    global COUNTER2
    COUNTER2 = 0
    def explore(self):
        global COUNTER2
        print('Exploring .. no' + str(COUNTER2))
        COUNTER2 += 1
        self.script.pick_position(False)
        self.script.pick_orientation(False)
        self.script.set_gripper_position()
        self.script.set_gripper_orientation()
        self.script.move_down()
        self.script.close_hand()
        self.script.lift_arm()
        label = self.script.successful_grasp()
        self.script.new_episode(label)
        self.robot.set_initial_position()

    global COUNTER3
    COUNTER3 = 600
    def collect_training_data(self):
        global COUNTER3
        self.start_vrep()
        choices = np.random.uniform(0, 1, 100000)
        for i in range(1, 100000):

            if choices[i] > 0.7: #exploit
                self.exploit()
            else:
                self.explore()

            if i%500 == 0:
                self.start_vrep(True)

            self.robot.delete_texture()
            self.robot.domain_randomize()

            if i%100 == 0:
                self.robot.delete_object()
                self.robot.delete_texture()
                self.robot.add_object()
                self.robot.reset_object_state()
                continue

            self.robot.reset_object_state()

# policy = Policy()
# policy.trainer()
