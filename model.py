import torch
import torch.nn.functional as F
from torch import nn 
from mdn import MDN


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


class ZhangNet(nn.Module):
    def __init__(self, im_h, im_w):
        super(ZhangNet, self).__init__()
        self.rgb_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.depth_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2)

        self.cn_1 = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=1, stride=1)
        self.cn_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.cn_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        c_out_h, c_out_w = out_size_cnns((im_h, im_w), [self.rgb_conv, self.cn_1, self.cn_2, self.cn_3])
        flattened_dims = c_out_h * c_out_w * self.cn_3.out_channels

        hidden_dim = 50
        image_encoding_dim = 64
        ee_dim = 6
        past_hist_dim = 5 * ee_dim

        self.ff_enc = nn.Sequential(
            nn.Linear(flattened_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_encoding_dim),
            nn.ReLU()
        )

        self.ff_aux = nn.Sequential(
            nn.Linear(image_encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ee_dim),
        )

        self.ff_out = nn.Sequential(
            nn.Linear(image_encoding_dim + past_hist_dim + ee_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ee_dim)
        )

    def forward(self, rgb_ims, depth_ims, past_poses):
        rgb_enc = F.relu(self.rgb_conv(rgb_ims))
        depth_enc = F.relu(self.depth_conv(depth_ims))

        combined_im = torch.cat((rgb_enc, depth_enc), dim=1)

        c_out = F.relu(self.cn_1(combined_im))
        c_out = F.relu(self.cn_2(c_out))
        c_out = F.relu(self.cn_3(c_out))

        flattened_im = torch.flatten(c_out, 1)
        flattened_past = torch.flatten(past_poses, 1)

        img_enc = self.ff_enc(flattened_im)
        aux_out = self.ff_aux(img_enc)

        full_encoding = torch.cat((img_enc, flattened_past, aux_out), dim=1)
        output = self.ff_out(full_encoding)
        return output, aux_out


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
        self.conv1_bn = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3_bn = nn.BatchNorm2d(self.conv3.out_channels)
        # self.drop_layer = nn.Dropout()


        print("Kernel Size: {}".format(self.conv1.kernel_size))
        o_height, o_width = output_size(image_height, image_width, self.conv1.kernel_size[0], stride=2)
        o_height, o_width = output_size(o_height, o_width, self.conv2.kernel_size[0], stride=2)
        o_height, o_width = output_size(o_height, o_width, self.conv3.kernel_size[0], stride=2)

        linear_input_size = o_height * o_width * self.conv3.out_channels

        self.linear1 = nn.Linear(linear_input_size, 32)
        self.linear1_bn = nn.BatchNorm1d(self.linear1.out_features)

        self.linear2 = nn.Linear(32 + joint_dim, 32 + joint_dim)
        self.linear3 = nn.Linear(32 + joint_dim, joint_dim)


    def forward(self, img_ins, pose_ins):
        conv1_out = F.leaky_relu(self.conv1_bn(self.conv1(img_ins)))
        conv2_out = F.leaky_relu(self.conv2_bn(self.conv2(conv1_out)))
        conv3_out = F.leaky_relu(self.conv3_bn(self.conv3(conv2_out)))

        # conv1_out = F.leaky_relu(self.conv1(img_ins))
        # conv2_out = F.leaky_relu(self.conv2(conv1_out))
        # conv3_out = F.leaky_relu(self.conv3(conv2_out))


        flattened_conv = torch.flatten(conv3_out, 1)

        lin1_out = F.leaky_relu(self.linear1_bn(self.linear1(flattened_conv)))
        # lin1_out = F.leaky_relu(self.linear1(flattened_conv))
        # lin1_out = self.drop_layer(lin1_out)

        image_and_pos = torch.cat((lin1_out, pose_ins), dim=1)

        lin2_out = F.leaky_relu(self.linear2(image_and_pos))

        output = self.linear3(lin2_out)

        return output


class ImageOnlyNet(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(ImageOnlyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv1_bn = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3_bn = nn.BatchNorm2d(self.conv3.out_channels)

        print("Kernel Size: {}".format(self.conv1.kernel_size))
        o_height, o_width = out_size_cnns((image_height, image_width), [self.conv1, self.conv2, self.conv3])

        linear_input_size = o_height * o_width * self.conv3.out_channels

        hidden_layer_dim = 100

        self.linear1 = nn.Linear(linear_input_size, hidden_layer_dim)
        self.lin_drop = nn.Dropout()

        self.linear2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.linear3 = nn.Linear(hidden_layer_dim, joint_dim)

    def forward(self, img_ins):
        conv1_out = F.leaky_relu(self.conv1_bn(self.conv1(img_ins)))
        conv2_out = F.leaky_relu(self.conv2_bn(self.conv2(conv1_out)))
        conv3_out = F.leaky_relu(self.conv3_bn(self.conv3(conv2_out)))

        flattened_conv = torch.flatten(conv3_out, 1)

        lin1_out = self.lin_drop(F.leaky_relu(self.linear1(flattened_conv)))
        lin2_out = F.leaky_relu(self.linear2(lin1_out))
        output = self.linear3(lin2_out)

        return output

def out_size_cnns(img_dims, cnns):
    current_dims = img_dims
    for cnn in cnns:
        current_dims = output_size(current_dims[0], current_dims[1], cnn.kernel_size[0], cnn.stride[0], cnn.padding[0])
    return current_dims

def output_size(in_height, in_width, kernel_size, stride=1, padding=0):
    out_height = int((in_height - kernel_size + padding * 2) / stride) + 1
    out_width = int((in_width - kernel_size + padding * 2) / stride) + 1
    return (out_height, out_width)


class ImageOnlyMDN(nn.Module):
    def __init__(self, image_height, image_width, joint_dim, mix_num):
        super(ImageOnlyMDN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)
        self.conv1_bn = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2)
        self.conv3_bn = nn.BatchNorm2d(self.conv3.out_channels)

        print("Kernel Size: {}".format(self.conv1.kernel_size))
        o_height, o_width = out_size_cnns((image_height, image_width), [self.conv1, self.conv2, self.conv3])
        linear_input_size = o_height * o_width * self.conv3.out_channels
        hidden_layer_dim = 512

        self.mdn = MDN(linear_input_size, hidden_layer_dim, joint_dim, mix_num)


    def forward(self, img_ins):
        conv1_out = F.leaky_relu(self.conv1_bn(self.conv1(img_ins)))
        conv2_out = F.leaky_relu(self.conv2_bn(self.conv2(conv1_out)))
        conv3_out = F.leaky_relu(self.conv3_bn(self.conv3(conv2_out)))

        flattened_conv = torch.flatten(conv3_out, 1)

        mu, std, pi = self.mdn(flattened_conv)

        return mu, std, pi


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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def setup_joints_model(device, joint_names):
    model = JointsNet(len(joint_names))
    model.to(device)
    print(model)
    return model
