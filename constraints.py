import numpy as np
import torch
import torch.nn.functional as F
from domains import *
from helper_funcs.rm import forward_kinematics
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
    def __init__(self, net, lower_bounds, upper_bounds, dt=1.0, use_cuda=True):
        self.net = net
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.use_cuda = use_cuda
        self.name = 'joint limits'
        self.dt = dt
        self.n_gvars = 0

    def params(self):
        return {"lower_bound": self.lower_bounds,
                "upper_bound": self.upper_bounds,
                "dt": self.dt}


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


class EndEffectorPosConstraint(Constraint):
    def __init__(self, net, min_y, joint_param_names, robot_model, dt=1.0, use_cuda=True):
        self.net = net
        self.use_cuda = use_cuda
        self.name = 'end effector pos constraint'
        self.robot_model = robot_model
        self.joint_param_names = joint_param_names
        self.min_y = min_y
        self.dt = dt
        self.n_gvars = 0

    def params(self):
        return {"min_y": self.min_y,
                "dt": self.dt}


    def get_condition(self, z_inp, z_out, input_batches, target_batches): 
        assert len(input_batches) == 2 # Should be passing in an (img_batch, pose_batch) pair here
        _, pose_ins = input_batches
        est_vels = self.net(*input_batches)

        next_pose_ests = pose_ins + self.dt * est_vels
        end_effector_positions = torch.tensor([forward_kinematics(np_est, self.joint_param_names, self.robot_model) for np_est in next_pose_ests])
        end_effector_ys = end_effector_positions[:,1]
        
        return dl2.GEQ(end_effector_ys, self.min_y)