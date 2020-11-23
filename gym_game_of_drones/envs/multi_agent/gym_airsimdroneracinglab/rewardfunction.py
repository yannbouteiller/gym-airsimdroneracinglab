import airsimdroneracinglab as airsim
from enum import Enum, auto
import copy
import numpy as np

# DEBUG workaround as GetObjectPose is bugged:
import math
MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = 100


DEFAULT_CONFIG = {
    'constant_penalty': -1.0,  # constant penalty per time-step
    'collision_radius': 0.5,  # collision with opponent
    'velocity_gain': 10.0,  # not real velocity: difference of distance to next objective between 2 get_reward()
    'gate_crossed_reward': 100.0,
    'gate_missed_penalty': -100.0,
    'collision_penatly': -10,  # collision with environment
    'death_penalty': -500,  # collision with opponent
    'death_constant_penalty': 0.0,  # after collision with opponent until the end of track (should be at least lower than constant summed penalty when lagging behind and not moving to avoid reward hacking)
    'end_of_track_bonus': 100.0,  # only when the last gate is crossed
    'lag_penalty': -0.5,  # constant additional penalty if not leading the way
    'kill_reward': 50.0,
    'gate_facing_reward_gain': 1.0
}


def quaternion_to_yaw(q):
    # yaw (z-axis rotation)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def quaternions_to_gate_yaw_cos(q1, q2):
    q_diff = q1.inverse() * q2
    yaw = quaternion_to_yaw(q_diff)
    return np.cos(yaw + np.pi / 2.0)


# enums:
class GateStatus(Enum):
    NOT_CROSSED_NOR_PASSED = auto()
    CROSSED = auto()
    PASSED = auto()


class Gate(object):
    """
    Gate
    """
    def __init__(self, gate_pose_dims_tuple):
        self.gate_pose = gate_pose_dims_tuple[0]
        self.gate_half_dims = gate_pose_dims_tuple[1]
        self.facing_vector = self.get_gate_facing_vector_from_quaternion(self.gate_pose.orientation)
        self.width_constraint_unit_vector = self.rotate_vector_by_quaternion(airsim.Vector3r(1.0, 0.0, 0.0), self.gate_pose.orientation)
        self.height_constraint_unit_vector = self.rotate_vector_by_quaternion(airsim.Vector3r(0.0, 0.0, 1.0), self.gate_pose.orientation)

    @staticmethod
    def rotate_vector_by_quaternion(vect, quat):
        res = quat * vect.to_Quaternionr() * quat.inverse()
        return airsim.Vector3r(res.x_val, res.y_val, res.z_val)

    @staticmethod
    def get_gate_facing_vector_from_quaternion(quat):
        return Gate.rotate_vector_by_quaternion(airsim.Vector3r(0.0, 1.0, 0.0), quat)

    def next_gate_status(self, start_position, end_position, eps=1e-5):
        """
        Checks whether the line defined from start_position to end_position has crossed the gate
        !: eps is used for numeric stability, a gate is not considered crossed if the dot product between the direction and the facing vector is < eps (even if it is actually crossed)
        """
        dot1 = (end_position - start_position).dot(self.facing_vector)
        if dot1 < eps:
            # print("DEBUG: wrong direction")
            return GateStatus.NOT_CROSSED_NOR_PASSED
        else:
            d = (self.gate_pose.position - start_position).dot(self.facing_vector) / dot1
            if not (0 <= d < 1):  # plane not crossed
                # print("DEBUG: plane not crossed")
                return GateStatus.NOT_CROSSED_NOR_PASSED
            else:  # is the intersection within the gate?
                intersection_position = start_position + (end_position - start_position) * d
                vector_from_center = intersection_position - self.gate_pose.position
                c1 = (abs(self.width_constraint_unit_vector.dot(vector_from_center)) < self.gate_half_dims.x_val)  # half width
                c2 = (abs(self.height_constraint_unit_vector.dot(vector_from_center)) < self.gate_half_dims.z_val)  # half height
                if c1 and c2:
                    # print("DEBUG: gate crossed: ", abs(self.width_constraint_unit_vector.dot(vector_from_center)), "<", self.gate_half_dims.x_val, " and ", abs(self.height_constraint_unit_vector.dot(vector_from_center)), "<", self.gate_half_dims.z_val)
                    return GateStatus.CROSSED
                else:
                    # print("DEBUG: gate missed: ", abs(self.width_constraint_unit_vector.dot(vector_from_center)), ">", self.gate_half_dims.x_val, " or ", abs(self.height_constraint_unit_vector.dot(vector_from_center)), ">", self.gate_half_dims.z_val)
                    return GateStatus.PASSED


class RewardFunction(object):
    """
    Reward function
    get_reward() has to be called every time step
    For multi-agent support, self.opponent_RewardFunction has to be set with set_opponent_RewardFunction()
    """
    def __init__(self, airsim_client, vehicle_name, base_offset, objectives=None, param_dict=DEFAULT_CONFIG):
        """
        Client must be an instanciated airsim client
        objectives is the return of get_ground_truth_gate_poses_and_half_dims(), it should be saved and passed to the constructor after airsim reset() when re-instantiating a RewardFunction
        param_dict is a dictionary that contains all the parameters of the reward function. It can be modified from the DEFAULT_CONFIG constant defined in this script
        """
        self.opponent_RewardFunction = None
        self.opponent_position = None
        self.done = False
        self.base_offset = base_offset
        self.airsim_client = airsim_client
        self.drone_name = vehicle_name
        if not objectives:
            self.objectives, self.gates_names = self.get_ground_truth_gate_poses_and_half_dims()
        else:
            self.objectives = objectives
            self.gates_names = None
        self.current_objective_idx = 0
        self.nb_crossed_gates = 0
        self.current_objective = Gate(self.objectives[self.current_objective_idx])
        self.current_kinematics = self.airsim_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
        self.current_position = self.current_kinematics.position + self.base_offset
        self.current_distance = self.current_position.distance_to(self.current_objective.gate_pose.position)
        self.current_collision_time_stamp = self.airsim_client.simGetCollisionInfo(vehicle_name=self.drone_name).time_stamp
        self.last_position = self.current_position
        self.last_distance = self.current_distance
        self.last_collision_time_stamp = self.current_collision_time_stamp
        self.objective_status = GateStatus.NOT_CROSSED_NOR_PASSED
        self.track_complete = False  # True when a track_complete reward is to be claimed
        # self.pending_death = False  # True when the drone has been killed during current time step
        self.death = False  # True when the drone is dead
        self.kill = False  # True the opponent is dead
        self.constant_penalty = param_dict['constant_penalty']
        self.collision_radius = param_dict['collision_radius']
        self.velocity_gain = param_dict['velocity_gain']
        self.gate_crossed_reward = param_dict['gate_crossed_reward']
        self.gate_missed_penalty = param_dict['gate_missed_penalty']
        self.collision_penatly = param_dict['collision_penatly']
        self.death_penalty = param_dict['death_penalty']
        self.death_constant_penalty = param_dict['death_constant_penalty']
        self.end_of_track_bonus = param_dict['end_of_track_bonus']
        self.lag_penalty = param_dict['lag_penalty']
        self.kill_reward = param_dict['kill_reward']
        self.gate_facing_reward_gain = param_dict['gate_facing_reward_gain']

    def get_objectives(self):
        """
        returns a hard copy of self.objectives to be passed to the contructor later
        also returns the sorted list of gates names
        This must be used only once to retrieve initial gate positions after creating the first RewardFunction, because gates_names will be None afterward
        """
        return copy.deepcopy(self.objectives), self.gates_names

    def set_opponent_RewardFunction(self, opponent_RewardFunction):
        """
        Use this function after creating both reward functions in the multi agent setting
        This is needed because AirSim uses the starting position of each drone as the origin of its local basis
        """
        self.opponent_RewardFunction = opponent_RewardFunction
        self.opponent_position = self.opponent_RewardFunction.current_position

    def switch_to_next_objective(self):
        """
        switches to the next objective
        returns True if race is finished
        returns False otherwise
        !: changes self.last_distance to the distance to new objective
        """
        self.current_objective_idx += 1
        if self.current_objective_idx >= len(self.objectives):  # track complete
            self.current_distance = 0.0
            return True
        else:
            self.current_objective = Gate(self.objectives[self.current_objective_idx])
            self.last_distance = self.last_position.distance_to(self.current_objective.gate_pose.position)
            return False

    def get_ground_truth_gate_poses_and_half_dims(self):
        """
        Returns a list of lists
        each list contains the pose and half dimensions of a gates
        one list per gate (sorted from first to last gate)
        """
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        assert len(gate_names_sorted_bad) > 0, "ERROR: the gate list returned by simListSceneObjects is empty"
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k])
        gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        nominal_inner_dims = self.airsim_client.simGetNominalGateInnerDimensions()
        res = []
        for gate_name in gate_names_sorted:
            p = self.airsim_client.simGetObjectPose(object_name=gate_name)
            s = self.airsim_client.simGetObjectScale(object_name=gate_name)
            # print(f"DEBUG: gate {gate_name}")
            # print(f"DEBUG: scale {gate_name}: {s}")
            cpt = 0
            while (math.isnan(p.position.x_val) or math.isnan(p.position.y_val) or math.isnan(p.position.z_val) or math.isnan(s.x_val) or math.isnan(s.y_val) or math.isnan(s.z_val)) and cpt < MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS:
                print(f"DEBUG: p for {gate_name} is {p}")
                print(f"DEBUG: s for {gate_name} is {s}")
                print(f"DEBUG: {gate_name} is nan, retrying...")
                cpt += 1
                p = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(p.position.x_val), f"ERROR: {gate_name} p.position.x_val is still {p.position.x_val} after {cpt} trials"
            assert not math.isnan(p.position.y_val), f"ERROR: {gate_name} p.position.y_val is still {p.position.y_val} after {cpt} trials"
            assert not math.isnan(p.position.z_val), f"ERROR: {gate_name} p.position.z_val is still {p.position.z_val} after {cpt} trials"
            assert not math.isnan(s.x_val), f"ERROR: {gate_name} p.position.x_val is still {s.x_val} after {cpt} trials"
            assert not math.isnan(s.y_val), f"ERROR: {gate_name} p.position.y_val is still {s.y_val} after {cpt} trials"
            assert not math.isnan(s.z_val), f"ERROR: {gate_name} p.position.z_val is still {s.z_val} after {cpt} trials"
            res.append([p, airsim.Vector3r(s.x_val * nominal_inner_dims.x_val / 2, s.y_val * nominal_inner_dims.y_val / 2, s.z_val * nominal_inner_dims.z_val / 2)])
        assert len(res) > 0, "ERROR: the final gates list is empty"
        return res, gate_names_sorted
        # return [self.airsim_client.simGetObjectPose(gate_name) for gate_name in gate_names_sorted]

    def update_state(self):
        """
        This function must be called before every call to get_reward()
        In the multi-agent setting, states of both drones must be updated before calling get_reward() for any drone
        """
        self.last_position = self.current_position
        self.last_distance = self.current_distance
        self.last_collision_time_stamp = self.current_collision_time_stamp
        self.current_kinematics = self.airsim_client.simGetGroundTruthKinematics(vehicle_name=self.drone_name)
        self.current_position = self.current_kinematics.position + self.base_offset
        self.current_collision_time_stamp = self.airsim_client.simGetCollisionInfo(vehicle_name=self.drone_name).time_stamp
        # print("DEBUG: simGetCollisionInfo:", self.airsim_client.simGetCollisionInfo(vehicle_name=self.drone_name))
        # self.pending_death = self.airsim_client.simIsRacerDisqualified(vehicle_name=self.drone_name)
        self.objective_status = self.current_objective.next_gate_status(self.last_position, self.current_position)
        if self.objective_status == GateStatus.CROSSED or self.objective_status == GateStatus.PASSED:
            if self.switch_to_next_objective():  # if track is finished (changes self.last_distance)
                self.track_complete = True
        self.current_distance = self.current_position.distance_to(self.current_objective.gate_pose.position)

    def get_reward(self):  # beware, this should not exceed -1.0 * constant-penalty (reward hacking)
        """
        We want to get high reward when we rush toward the next objective (continuous reward, useful for training)
        We want to pass through the gate in the right direction (event reward)
        We want to get there as fast as possible (negative constant reward)
        We want to avoid collision with the environement (event negative reward)
        We want to avoid breaking the safety radius with the leading drone (linear negative reward if within the safety radius) (removed)
        We want to avoid collision with the leading drone (event negative reward and end of the episode if within the collision radius)
        We want to lead the way (binary negative reward if lagging behind)
        We want the opponent to break the safety radius when we lead the way (affine negative reward when in the multiagent setting) (removed)
        We want to kill the opponent (event reward)

        If we killed the opponent, we switch back to single agent setting
        (opponent-based rewards are penalties to avoid reward hacking)
        This conceptually allows us to train the same network for both settings

        !: For safety, we should ensure that death/kill is detected at the same time for both drones
        (this should not be a problem if the simulator is paused when update_state() is called for both drones)
        """
        if self.death:
            return self.death_constant_penalty
        elif self.done:  # track_complete
            return 0.0

        # environment collision:
        if self.current_collision_time_stamp == self.last_collision_time_stamp:
            col_rew = 0.0
        else:
            # print('DEBUG: collision detected')
            col_rew = self.collision_penatly

        # current gate passed:
        if self.objective_status == GateStatus.NOT_CROSSED_NOR_PASSED:
            gate_rew = 0.0
        else:
            if self.objective_status == GateStatus.CROSSED:
                # print("DEBUG: ", self.drone_name, " crossed a gate")
                self.nb_crossed_gates += 1
                gate_rew = self.gate_crossed_reward
                if self.track_complete:
                    # print("DEBUG: ", self.drone_name, " gets end of track bonus")
                    gate_rew += self.end_of_track_bonus
            else:  # maybe better to check all gates for pass?
                # print("DEBUG: ", self.drone_name, " missed a gate")
                gate_rew = self.gate_missed_penalty
            if self.track_complete:
                # print("DEBUG: ", self.drone_name, " completed the track")
                self.track_complete = False
                self.done = True
            self.objective_status = GateStatus.NOT_CROSSED_NOR_PASSED

        # velocity toward objective:
        # print("DEBUG: ", self.drone_name, " distance to gate: ", self.current_distance)
        distance_difference = self.last_distance - self.current_distance

        # gate-facing reward:
        rew_gate_facing = self.gate_facing_reward_gain * quaternions_to_gate_yaw_cos(self.current_kinematics.orientation, self.current_objective.gate_pose.orientation)
        # print("DEBUG: rew_gate_facing:", rew_gate_facing)

        rew = self.constant_penalty + distance_difference * self.velocity_gain + col_rew + gate_rew + rew_gate_facing

        if self.opponent_RewardFunction is None or self.kill:  # single agent rewards
            # print("--- DEBUG: single agent setting ---")
            return rew
        else:  # multi agent rewards
            # print("--- DEBUG: multi agent setting ---")
            opp_rew = 0.0
            self.opponent_position = self.opponent_RewardFunction.current_position
            distance_to_opponent = self.current_position.distance_to(self.opponent_position)
            # kill = self.opponent_RewardFunction.pending_death
            # death = self.pending_death
            # print("DEBUG: ", self.drone_name, " distance to opponent: ", distance_to_opponent)
            # if (self.opponent_RewardFunction.nb_crossed_gates > self.nb_crossed_gates) or (self.opponent_RewardFunction.nb_crossed_gates == self.nb_crossed_gates and self.current_distance >= self.opponent_RewardFunction.current_distance):
            if (self.opponent_RewardFunction.current_objective_idx > self.current_objective_idx) or (self.opponent_RewardFunction.current_objective_idx == self.current_objective_idx and self.current_distance >= self.opponent_RewardFunction.current_distance):
                # we are behind the opponent, and therefore we must avoid avoid collisions
                # print("DEBUG: ", self.drone_name, " is behind")
                opp_rew += self.lag_penalty
                if distance_to_opponent <= self.collision_radius:
                # if death:
                    # print("DEBUG: death of ", self.drone_name)
                    self.death = True
                    opp_rew += self.death_penalty
                    self.done = True
            else:
                # print("DEBUG: ", self.drone_name, " is leading")
                # we are leading the way
                if distance_to_opponent <= self.collision_radius:
                # if kill:
                    # print("DEBUG: kill performed by ", self.drone_name)
                    self.kill = True
                    opp_rew += self.kill_reward
            # assert not math.isnan(rew), f"DEBUG: multiagent rew is {rew}"
            # assert not math.isnan(opp_rew), f"DEBUG: multiagent opp_rew is {opp_rew}"
            return rew + opp_rew
