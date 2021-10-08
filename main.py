import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import laspy


import torch.nn as nn
import torch.nn.functional as F
import torch

import os.path as osp

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import intersection_and_union as i_and_u
from PIL import Image
from pointnet2_classification import SAModule, GlobalSAModule, MLP
from torch_geometric.data import  Data, Batch

from torchvision.utils import save_image

import glob
import open3d as o3d
import torchvision.transforms as transforms
import rasterio
import numpy as np
# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    SoftPhongShader,
    FoVPerspectiveCameras
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="mydata_point", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)


cuda = True if torch.cuda.is_available() else False
class ImgtoPointDataset(data.Dataset):

    ATTR_EXLUSION_LIST = ['X', 'Y', 'Z','red', 'green', 'blue',
                          'flag_byte', 'scan_angle_rank', 'user_data',
                          'pt_src_id', 'gps_time']
    ATTR_EXTRA_LIST = ['num_returns']
    def __init__(self, root, npoints=20000, transform=None):
        self.npoints = npoints
        self.root = root
        self.pointlist = []
        self.rgblist = []
        self.datalist = []
        
        self.transform = transform
        self.pointpath = root + "/pointcloud_path/"
        print(self.pointpath )
        self.point_list  = glob.glob(self.pointpath + "/*.las")[:]
        print(self.point_list )
        count = 0
        for file in self.point_list:
            print(file)
            # point cloud 取得
            file_h = laspy.file.File(file, mode='r')
            print(file_h.header.min[0])
            print(file_h.header.min[2])
            print(file_h.header.min[1])
            src = np.vstack([file_h.x, file_h.y, file_h.z]).transpose()
            if(len(src)<npoints):continue
            rgb = np.vstack([file_h.red, file_h.green, file_h.blue]).transpose()
            rgb = rgb/255.0
            print(np.amin(rgb, axis=0))
            print(np.amax(rgb, axis=0))
            points = file_h.points['point']
            attr_names = [a for a in points.dtype.names] + ImgtoPointDataset.ATTR_EXTRA_LIST
            features = np.array([getattr(file_h, name) for name in attr_names
                                if name not in ImgtoPointDataset.ATTR_EXLUSION_LIST]).transpose()
            
            print(features[:,1])
            features = features/1.0
            names = [name for name in attr_names if name not in ImgtoPointDataset.ATTR_EXLUSION_LIST]
            print(names)

            file_h.close()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(src)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
            pcd = pcd.select_down_sample(ind)
            src = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            normlized_xyz = np.zeros((npoints, 3))
            normlized_rgb = np.zeros((npoints, 3))
            normlized_feature = np.zeros((npoints, 3))
            self.coord_min, self.coord_max = np.amin(src, axis=0)[:3], np.amax(src, axis=0)[:3]
            
            if(self.coord_max[0]==0):continue
            if(self.coord_max[1]==0):continue
            if(self.coord_max[2]==0):continue
            print(np.amin(src, axis=0)[:3] )
            print(np.amax(src, axis=0)[:3] )
            src[:, 0] = ((src[:, 0] - self.coord_min[0])/30.0) - 0.5
            src[:, 1] = ((src[:, 1] - self.coord_min[1])/30.0) - 0.5
            src[:, 2] = ((src[:, 2] - self.coord_min[2])/30.0) 
            features[:,0] = features[:,0]/ 4000.0 #'intensity', 'raw_classification', 'num_returns']
            features[:,1] = features[:,1]/ 17.0
            features[:,2] = features[:,2]/ 8.0


            print(np.amin(src, axis=0)[:3] )
            print(np.amax(src, axis=0)[:3] )
            if(len(src) >=npoints):
                normlized_xyz[:,:]=src[:npoints,:]
                normlized_rgb[:,:]=rgb[:npoints,:]
                normlized_feature[:,:] = features[:npoints,:]
            else:
                normlized_xyz[:len(src),:]=src[:,:]

            self.pointlist.append(normlized_xyz)
            self.rgblist.append(normlized_rgb)
            normlized_xyz = torch.from_numpy(normlized_xyz).float()
            random_features = torch.randn(npoints,6)
            random_features[:, :3] = torch.from_numpy(normlized_feature).float()

            self.datalist.append(Data(pos=normlized_xyz[:, :], x=random_features[ :, :3]))

                

        self.data_num = len(self.pointlist)
        

    def __getitem__(self, index):
        rgb = self.rgblist[index]
        rgb = torch.from_numpy(rgb)
        point = self.pointlist[index]
        point = torch.from_numpy(point)
        geom = self.datalist[index]
        return rgb, point, geom

    def __len__(self):
        return len(self.pointlist)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])
train_dataset = ImgtoPointDataset(root="/path/to/root/",transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                          num_workers=16)

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image

lambda_pixel = 100
lambda_GAN = 1
lambda_IMAGE = 1
lambda_POINT = 1

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.5, 0.1, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.5, 0.5, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        # print(sa1_out[0].size())
        sa2_out = self.sa2_module(*sa1_out)
        # print(sa2_out[0].size())
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = F.sigmoid(x)
        return x

class Net_D(torch.nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.sa1_module = SAModule(0.5, 0.1, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.5, 0.5, MLP([128 + 3, 128, 128, 256]))

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        # print(sa1_out[0].size())
        x, _, _  = self.sa2_module(*sa1_out)
        # print(sa2_out[0].size())

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return x

generator = Net(3).to(device)
pouintD = Net_D().to(device)
##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels + 1, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    pouintD = pouintD.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    pouintD.load_state_dict(torch.load("saved_models/%s/pouintD_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_PD = torch.optim.Adam(pouintD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(real_A, real_B, fake_B):
    """Saves a generated sample from the validation set"""
    save_image(real_A, "images/%s/%s_realA.png" % (opt.dataset_name, batches_done), normalize=True)
    save_image(real_B, "images/%s/%s_realB.png" % (opt.dataset_name, batches_done), normalize=True)
    save_image(fake_B, "images/%s/%s_fakeB.png" % (opt.dataset_name, batches_done), normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()
optimizer_G.zero_grad()
optimizer_PD.zero_grad()
optimizer_D.zero_grad()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):

        # Model inputs
        real_RGB  = batch[0].float() # real_RGB
        real_RGB = real_RGB.to(device)
        point  = batch[1].float()
        point = point.to(device)
        geom = batch[2]
        geom = geom.to(device)
        # Initialize a camera.
        R, T = look_at_view_transform(1, 0, 0) #真上
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0)
        raster_settings = PointsRasterizationSettings(
            image_size=512, 
            radius = 0.01,
            points_per_pixel = 10
        )


        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )
        verts = Pointclouds(points=[point[0,:,:]])
        real_A = rasterizer(verts)
        real_A = real_A[1]
        # print(real_A.size())
        real_A = real_A[:,128:256,128:256,:1]
        # print(real_A.size())
        real_A = real_A.transpose(1,3).transpose(2,3)
        # print(real_A.size())


    
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
        valid_P = Variable(Tensor(np.ones((1,5000,1))), requires_grad=False)
        fake_P = Variable(Tensor(np.zeros((1,5000,1))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        

        # GAN loss
        fake_RGB = generator(geom)
        point = point[0,:,:]
        # print(point.size())
        #Rendering
        point_cloud_fake = Pointclouds(points=[point], features=[fake_RGB])
        point_cloud_real = Pointclouds(points=[point], features=[real_RGB[0,:,:]])
        fake_RGB_image = renderer(point_cloud_fake)
        real_RGB_image = renderer(point_cloud_real)
        print(fake_RGB_image.size())
        print(real_RGB_image.size())
        # print(img.size())

        fake_RGB_image = fake_RGB_image[:,128:256,128:256,:]
        real_RGB_image = real_RGB_image[:,128:256,128:256,:]
        
        # print(fake_B.size())
        fake_RGB_image = fake_RGB_image.transpose(1,3).transpose(2,3)
        real_RGB_image = real_RGB_image.transpose(1,3).transpose(2,3)
        # print(fake_B.size())


        # image based D
        pred_fake = discriminator(fake_RGB_image, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid) * lambda_IMAGE
        # point based D
        point_cloud_fake = Data(pos=point, x=fake_RGB)
        point_cloud_fake = Batch.from_data_list([point_cloud_fake])
        print(point_cloud_fake)        
        pred_fake_point = pouintD(point_cloud_fake)
        loss_GAN_point = criterion_GAN(pred_fake_point, valid_P) * lambda_POINT
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_RGB_image, real_RGB_image) * lambda_IMAGE
        # Point-wise loss
        loss_point = criterion_pixelwise(fake_RGB, real_RGB) * lambda_POINT

        # Total loss
        loss_G = lambda_GAN * loss_GAN + loss_GAN_point + lambda_pixel * loss_pixel + lambda_pixel * loss_point

        loss_G.backward()
        if (i+1)%1 == 0:
            optimizer_G.step()
            optimizer_G.zero_grad()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        

        # Real loss
        pred_real = discriminator(real_RGB_image, real_A)
        loss_real = criterion_GAN(pred_real, valid) * lambda_IMAGE
        point_cloud_real = Data(pos=point, x=real_RGB[0,:,:])
        point_cloud_real = Batch.from_data_list([point_cloud_real])
        print(point_cloud_real)
        pred_real_point = pouintD(point_cloud_real)
        loss_real_p = criterion_GAN(pred_real_point, valid_P) * lambda_POINT

        # Fake loss
        pred_fake = discriminator(fake_RGB_image.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake) * lambda_IMAGE
        point_cloud_fake = Data(pos=point, x=fake_RGB.detach())
        point_cloud_fake = Batch.from_data_list([point_cloud_fake])        
        pred_fake_point = pouintD(point_cloud_fake)
        loss_fake_p = criterion_GAN(pred_fake_point, fake_P) * lambda_POINT

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake + loss_real_p + loss_fake_p) * lambda_GAN

        loss_D.backward()
        if (i+1)%1 == 0:
            optimizer_D.step()
            optimizer_D.zero_grad()
            optimizer_PD.step()
            optimizer_PD.zero_grad()


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i
        batches_left = opt.n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(real_A, real_RGB_image, fake_RGB_image)

        if (i+1)%100 == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(pouintD.state_dict(), "saved_models/%s/pouintD_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
