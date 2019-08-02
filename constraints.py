import numpy as np
import torch
import torch.nn.functional as F
from domains import Box
from helper_funcs.rm import forward_kinematics, forward_kinematics_orientation
import sys
sys.path.append('../../')
import dl2lib as dl2


def kl(p, log_q):
    return torch.sum(-p * log_q + p * torch.log(p), dim=1)


class Constraint:

    def eval_z(self, z_batches):
        if self.use_cuda:
            z_inputs = [torch.cuda.FloatTensor(z_batch) for z_batch in z_batches]
        else:
            z_inputs = [torch.FloatTensor(z_batch) for z_batch in z_batches]

        for z_input in z_inputs:
            z_input.requires_grad_(True)
        z_outputs = [self.net(z_input) for z_input in z_inputs]
        for z_out in z_outputs:
            z_out.requires_grad_(True)
        return z_inputs, z_outputs

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        assert False

    def loss(self, x_batches, y_batches, z_batches, args):
        if z_batches is not None:
            z_inp, z_out = self.eval_z(z_batches)
        else:
            z_inp, z_out = None, None

        constr = self.get_condition(z_inp, z_out, x_batches, y_batches)
        
        neg_losses = dl2.Negate(constr).loss()
        pos_losses = constr.loss()
        sat = constr.satisfy()
            
        return neg_losses, pos_losses, sat, z_inp


class JointLimitsConstraint(Constraint):
    def __init__(self, net, lower_bounds, upper_bounds, dt=0.05, use_cuda=True):
        self.net = net
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.use_cuda = use_cuda
        self.name = 'joint limits'
        self.dt = dt
        self.n_gvars = 1

    def params(self):
        return {"lower_bounds": self.lower_bounds,
                "upper_bounds": self.upper_bounds
                }


    def get_condition(self, z_inp, z_out, input_batches, target_batches): 
        assert len(input_batches) == 2 # Should be passing in an (img_batch, pose_batch) pair here
        _, pose_ins = input_batches
        est_vels = self.net(*input_batches)

        next_pose_est = pose_ins + self.dt * est_vels

        limit_conjunctions = []
        for i in range(len(self.lower_bounds)):
            if self.lower_bounds[i] is not None:
                limit_conjunctions.append(dl2.GEQ(next_pose_est[:, i], self.lower_bounds[i]))
                limit_conjunctions.append(dl2.LEQ(next_pose_est[:, i], self.upper_bounds[i]))

        return dl2.And(limit_conjunctions)


    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        # Ignore the actual joint inputs - this constraints is that you shouldn't violate joint limits *no matter what*
        n_batch = x_batches[0].size()[0]
        return [[Box(self.lower_bounds, self.upper_bounds) for i in range(n_batch)]]


class EndEffectorPosConstraint(Constraint):
    def __init__(self, net, min_y, joint_param_names, robot_model, eps=1e-3, dt=0.05, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda
        self.name = 'end effector pos constraint'
        self.robot_model = robot_model
        self.joint_param_names = joint_param_names
        self.min_y = min_y

        self.eps = eps
        self.dt = dt
        self.n_gvars = 1

    def params(self):
        return {"min_y": self.min_y}


    def get_condition(self, z_inp, z_out, input_batches, target_batches): 
        assert len(input_batches) == 2 # Should be passing in an (img_batch, pose_batch) pair here
        _, pose_ins = input_batches
        est_vels = self.net(*input_batches)

        next_pose_ests = pose_ins + self.dt * est_vels
        end_effector_positions = torch.tensor([forward_kinematics(np_est, self.joint_param_names, self.robot_model) for np_est in next_pose_ests], dtype=torch.float)
        end_effector_ys = end_effector_positions[:,1]
        
        return dl2.GEQ(end_effector_ys, self.min_y)

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        _, pose_ins = x_batches
        return [[Box(pose_ins[i] - self.eps, pose_ins[i] + self.eps)] for i in range(len(pose_ins))]


class StayInZone(Constraint):
    def __init__(self, net, min_bounds, max_bounds, joint_param_names, robot_model, eps=1e-3, dt=0.05, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda
        self.name = 'stay in zone'
        self.robot_model = robot_model
        self.joint_param_names = joint_param_names

        self.eps = eps
        self.dt = dt # TODO: Switch to positional movement?
        self.n_gvars = 1 ## TODO: Is this what we want for this?


        assert len(min_bounds) == len(max_bounds) == 3 # Bounds should be (x,y,z) coords
        self.min_vec = min_bounds
        self.max_vec = max_bounds

    def params(self):
        return {"min_bounds": self.min_bounds,
                "max_bounds": self.max_bounds
                }

        
    def get_condition(self, z_inp, z_out, input_batches, target_batches): 
        assert len(input_batches) == 2 # Should be passing in an (img_batch, pose_batch) pair here
        _, pose_ins = input_batches
        est_vels = self.net(*input_batches)

        next_pose_ests = pose_ins + self.dt * est_vels
        end_effector_positions = torch.FloatTensor([forward_kinematics(np_est, self.joint_param_names, self.robot_model) for np_est in next_pose_ests])
        
        return dl2.And([
            dl2.GEQ(end_effector_positions[:,0], self.min_bounds[0]),
            dl2.GEQ(end_effector_positions[:,1], self.min_bounds[1]),
            dl2.GEQ(end_effector_positions[:,2], self.min_bounds[2]),
            dl2.LEQ(end_effector_positions[:,0], self.max_bounds[0]),
            dl2.LEQ(end_effector_positions[:,1], self.max_bounds[1]),
            dl2.LEQ(end_effector_positions[:,2], self.max_bounds[2]),
        ])


    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        _, pose_ins = x_batches
        # TODO: Perhaps the domain for this could actually be everything within the bounds
        return [[Box(pose_ins[i] - self.eps, pose_ins[i] + self.eps)] for i in range(len(pose_ins))]


class MoveSlowly(Constraint):
    def __init__(self, net, speed_limit, joint_param_names, robot_model, eps=1e-3, dt=0.05, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda
        self.name = 'move slowly'
        self.robot_model = robot_model
        self.joint_param_names = joint_param_names

        self.eps = eps
        self.dt = dt
        self.n_gvars = 1

        self.speed_limit = speed_limit

    def params(self):
        return {"speed_limit": self.speed_limit}

        
    def get_condition(self, z_inp, z_out, input_batches, target_batches): 
        assert len(input_batches) == 2 # Should be passing in an (img_batch, pose_batch) pair here
        est_vels = self.net(*input_batches)

        velocity_displacements = self.dt * est_vels
        ee_disp = torch.FloatTensor([forward_kinematics(v_dis, self.joint_param_names, self.robot_model) for v_dis in velocity_displacements])
        
        return dl2.LEQ(ee_disp.norm(dim=1), self.speed_limit)


    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        _, pose_ins = x_batches
        return [[Box(pose_ins[i] - self.eps, pose_ins[i] + self.eps)] for i in range(len(pose_ins))]


class MatchOrientation(Constraint):
    def __init__(self, net, orientation, joint_param_names, robot_model, eps=1e-3, dt=0.05, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda
        self.name = 'match orientation'
        self.robot_model = robot_model
        self.joint_param_names = joint_param_names

        self.eps = eps
        self.dt = dt
        self.n_gvars = 1

        self.orientation = orientation
    

    def params(self):
        return {"orientation": self.orientation}

    
    def get_condition(self, z_inp, z_out, input_batches, target_batches):
        assert len(input_batches) == 2 # Should be passing in an (img_batch, pose_batch) pair here
        _, pose_ins = input_batches
        est_vels = self.net(*input_batches)

        next_pose_ests = pose_ins + self.dt * est_vels
        end_effector_orientation = torch.FloatTensor(
            [forward_kinematics_orientation([0.0, 0.0, 1.0, 1.0], np_est, self.joint_param_names, self.robot_model) 
             for np_est in next_pose_ests])
        
        return dl2.And([
            dl2.EQ((1.0 - F.cosine_similarity(self.orientation, end_effector_orientation)) / 2.0, 0.0)
        ])


    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        _, pose_ins = x_batches
        return [[Box(pose_ins[i] - self.eps, pose_ins[i] + self.eps)] for i in range(len(pose_ins))]


class SmoothMotion(Constraint):
    def __init__(self, net, max_dist, joint_param_names, robot_model, eps=1e-3, dt=0.05, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda
        self.name = 'smooth motion'
        self.robot_model = robot_model
        self.joint_param_names = joint_param_names

        self.eps = eps
        self.dt = dt
        self.n_gvars = 1

        self.max_dist = max_dist


    def params(self):
        return {"max_dist": self.max_dist}


    def get_condition(self, z_inp, z_out, input_batches, target_batches):

        next_pose_ests = z_inp + self.dt * z_out
        old_ee_pos = torch.FloatTensor(
            [forward_kinematics(p, self.joint_param_names, self.robot_model) for p in z_inp]
        )
        new_ee_pos = torch.FloatTensor(
            [forward_kinematics(p, self.joint_param_names, self.robot_model) for p in next_pose_ests]
        )
        
        return dl2.And([
            dl2.LEQ((old_ee_pos - new_ee_pos).norm(dim=1), self.max_dist)
        ])


    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        _, pose_ins = x_batches
        return [[Box(pose_ins[i] - self.eps, pose_ins[i] + self.eps)] for i in range(len(pose_ins))]

"""
Smooth motion 2 (Not gonna do this one)
Fit a cubic spline (or something similar) to a set of consecutive predictions.
Measure its smoothness over the relevant interval using the integral of the squared second derivative
Set a threshold on this for sets of 4 points...
"""
