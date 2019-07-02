import torch
import torch.nn.functional as F
from torch import nn 


class ImageAndJointsNet(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(ImageAndJointsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 7)
        self.conv2 = nn.Conv2d(3, 3, 5)
        self.conv3 = nn.Conv2d(3, 3, 3)
        self.drop_layer = nn.Dropout()

        o_height, o_width = output_size(image_height, image_width, 7)
        o_height, o_width = output_size(o_height, o_width, 5)
        o_height, o_width = output_size(o_height, o_width, 3)

        linear_input_size = o_height * o_width * 3

        self.linear1 = nn.Linear(linear_input_size, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32 + joint_dim, joint_dim)

    def forward(self, img_ins, pose_ins):
        conv1_out = F.relu(self.conv1(img_ins))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))

        flattened_conv = torch.flatten(conv3_out, 1)

        lin1_out = F.relu(self.linear1(flattened_conv))
        lin1_out = self.drop_layer(lin1_out)
        lin2_out = F.relu(self.linear2(lin1_out))

        image_and_pos = torch.cat((lin2_out, pose_ins), dim=1)
        output = self.linear3(image_and_pos)

        return output
        #return {"conv1_out": conv1_out, "conv2_out": conv2_out, "conv3_out": conv3_out, "output": output}


class JointsNet(nn.Module):
    def __init__(self, joint_dim):
        super(JointsNet, self).__init__()
        self.drop_layer = nn.Dropout()

        self.linear1 = nn.Linear(joint_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, joint_dim)

    def forward(self, pose_ins):

        lin1_out = F.relu(self.linear1(pose_ins))
        lin2_out = F.relu(self.linear2(lin1_out))
        output = self.linear3(lin2_out)

        return output


def output_size(in_height, in_width, kernel_size, stride=1, padding=0):
    out_height = int((in_height - kernel_size + padding * 2) / stride) + 1
    out_width = int((in_width - kernel_size + padding * 2) / stride) + 1
    return (out_height, out_width)


def setup_model(device, height, width, joint_names):
    model = ImageAndJointsNet(height, width, len(joint_names))
    model.to(device)
    print(model)  # If this isn't enough info, try the "pytorch-summary" package
    return model


def load_model(model_path, device, height, width, joint_names):
    model = ImageAndJointsNet(height, width, len(joint_names))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def setup_joints_model(device, joint_names):
    model = JointsNet(len(joint_names))
    model.to(device)
    print(model)
    return model