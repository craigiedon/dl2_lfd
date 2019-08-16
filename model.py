import torch
import torch.nn.functional as F
from torch import nn 


class SpatialSoftmax(nn.Module):
    def __init__(self, h, w):
        super(SpatialSoftmax, self).__init__()
        x_inds = torch.arange(w).repeat(h, 1).float()
        y_inds = torch.arange(h).repeat(w, 1).transpose(0,1).float()

        self.register_buffer("x_inds", x_inds)
        self.register_buffer("y_inds", y_inds)


    def forward(self, ins):
        n, c, h, w = ins.shape

        flat_maxed = F.softmax(ins.reshape(n, c, -1), dim=2)
        flat_maxed = flat_maxed.reshape(n, c, h, w)

        expected_xs = (flat_maxed * self.x_inds).sum(dim=(2,3))
        expected_ys = (flat_maxed * self.y_inds).sum(dim=(2,3))

        feature_points = torch.cat((expected_xs, expected_ys), dim=1)

        return feature_points


class LevineNet(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(LevineNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1)

        c1_h, c1_w = output_size(image_height, image_width, 7, stride=2)
        c2_h, c2_w = output_size(c1_h, c1_w, 5)
        c3_h, c3_w = output_size(c2_h, c2_w, 5)

        self.cout_h, self.cout_w = (c3_h, c3_w)

        self.spatial_sm = SpatialSoftmax(c3_h, c3_w)

        linear_input_size = 16 * 2 + joint_dim
        hidden_layer_width = 40

        self.linear1 = nn.Linear(linear_input_size, hidden_layer_width)
        self.linear2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.linear3 = nn.Linear(hidden_layer_width, joint_dim)

        self.aux_outputs = {}

    
    def forward(self, img_ins, pose_ins):
        conv1_out = F.relu(self.conv1(img_ins))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))

        feature_points = self.spatial_sm(conv3_out)

        full_input = torch.cat((feature_points, pose_ins), dim=1)

        l1_out = F.relu(self.linear1(full_input))
        l2_out = F.relu(self.linear2(l1_out))
        l3_out = self.linear3(l2_out)

        self.aux_outputs["conv3_out"] = conv3_out
        self.aux_outputs["feature_points"] = feature_points

        return l3_out

        
class ImageAndJointsNet(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(ImageAndJointsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.drop_layer = nn.Dropout()

        print("Kernel Size: {}".format(self.conv1.kernel_size))
        o_height, o_width = output_size(image_height, image_width, self.conv1.kernel_size[0], stride=2)
        o_height, o_width = output_size(o_height, o_width, self.conv2.kernel_size[0], stride=2)
        o_height, o_width = output_size(o_height, o_width, self.conv3.kernel_size[0], stride=2)

        linear_input_size = o_height * o_width * self.conv3.out_channels

        self.linear1 = nn.Linear(linear_input_size, 32)
        self.linear2 = nn.Linear(32 + joint_dim, 32 + joint_dim)
        self.linear3 = nn.Linear(32 + joint_dim, joint_dim)


    def forward(self, img_ins, pose_ins):
        conv1_out = F.leaky_relu(self.conv1(img_ins))
        conv2_out = F.leaky_relu(self.conv2(conv1_out))
        conv3_out = F.leaky_relu(self.conv3(conv2_out))

        flattened_conv = torch.flatten(conv3_out, 1)

        lin1_out = F.leaky_relu(self.linear1(flattened_conv))
        # lin1_out = self.drop_layer(lin1_out)

        image_and_pos = torch.cat((lin1_out, pose_ins), dim=1)

        lin2_out = F.leaky_relu(self.linear2(image_and_pos))

        output = self.linear3(lin2_out)

        return output


"""
class ImageOnlyNet(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(ImageOnlyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.drop_layer = nn.Dropout()

        print("Kernel Size: {}".format(self.conv1.kernel_size))
        o_height, o_width = output_size(image_height, image_width, self.conv1.kernel_size[0], stride=2)
        o_height, o_width = output_size(o_height, o_width, self.conv2.kernel_size[0], stride=2)
        o_height, o_width = output_size(o_height, o_width, self.conv3.kernel_size[0], stride=2)

        linear_input_size = o_height * o_width * self.conv3.out_channels

        hidden_units = 32
        self.linear1 = nn.Linear(linear_input_size, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, joint_dim)


    def forward(self, img_ins, pose_ins):
        conv1_out = F.relu(self.conv1(img_ins))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))

        flattened_conv = torch.flatten(conv3_out, 1)

        lin1_out = F.relu(self.linear1(flattened_conv))
        # lin1_out = self.drop_layer(lin1_out)
        lin2_out = F.relu(self.linear2(lin1_out))

        output = self.linear3(lin2_out)

        return output
"""


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
    model = ImageOnlyNet(height, width, len(joint_names))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def setup_joints_model(device, joint_names):
    model = JointsNet(len(joint_names))
    model.to(device)
    print(model)
    return model
