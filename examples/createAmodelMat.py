"""
    Creating a numpy arr B x 4 x H x W
    Given the camera's positions and a model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import soft_renderer as sr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


class Model(nn.Module):
    def __init__(self, template_path, scale=1.0, offset=np.zeros(3).astype(np.float32)):
        super(Model, self).__init__()

        # set template mesh
        self.template_mesh = sr.Mesh.from_obj(template_path)
        # originaly, before normalizing
        self.register_buffer('vertices', self.template_mesh.vertices * scale)  # originaly *0.5

        # Normalizing test
        # self.register_buffer('vertices', self.template_mesh.vertices * scale - torch.from_numpy(
        #                         offset[None, None, :]).cuda())  # originaly *0.5

        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
            # temp = torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))
            # temp = torch.from_numpy(np.array([+0.2, -0.0, +0.0]).astype(np.float32))
        temp = torch.from_numpy(offset[None, None, :])
        self.register_parameter('center', nn.Parameter(temp))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        self.volumeLoss = sr.VolumeLoss(self.faces[0].cpu())

    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))  # might need an epsilon
        # print(base)
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()
        volume_loss = self.volumeLoss(vertices).mean()

        return sr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss, volume_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cen', '--center', type=str,
                        default='0.0,0.0,0.0')
    parser.add_argument('-of', '--filename-output', type=str,
                        default='plane_shifted2')
    parser.add_argument('-c', '--camera-input', type=str,
                        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-m', '--object-mesh', type=str,
                        default=os.path.join(data_dir, 'obj/plane/two_planes_shifted2.obj'))
    # default = os.path.join(data_dir, 'obj/sphere/sphere_1352.obj'))
    # default = os.path.join(data_dir, 'obj/simple/cube.obj'))
    parser.add_argument('-od', '--output-dir', type=str,
                        default=os.path.join(data_dir, 'results/model_matrices'))
    parser.add_argument('-b', '--batch-size', type=int,
                        default=120)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    center = np.fromstring(args.center, dtype=float, sep=',')
    model = Model(args.object_mesh, scale=1.0, offset=center.astype(np.float32)).cuda()
    transform = sr.LookAt(viewing_angle=15)
    lighting = sr.Lighting(intensity_ambient=0.5, intensity_directionals=0.5)
    rasterizer = sr.SoftRasterizer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard')

    # read camera poses
    cameras = np.load(args.camera_input).astype('float32')

    # ------- Saving the input images
    # for count in range(images.shape[0]):
    #     image = images[count, -1].reshape(64, 64)
    #     imageio.imsave(os.path.join(args.output_dir, 'input_%05d.png' % count), (255 * image.astype(np.uint8)))

    clipping = False
    #animatingTransformation = True
    savingImageArray = True

    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))

    camera_distances = torch.from_numpy(cameras[:, 0])
    elevations = torch.from_numpy(cameras[:, 1])
    viewpoints = torch.from_numpy(cameras[:, 2])
    transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    mesh, _, _, _ = model(args.batch_size)

    # render
    mesh = lighting(mesh)
    mesh = transform(mesh)
    images_pred = rasterizer(mesh)

    if savingImageArray:
        imageArr = (images_pred.detach().cpu().numpy() * 255).astype(np.uint8)
        np.save(os.path.join(args.output_dir, args.filename_output+'['+args.center+'].npy'), imageArr)

    print('Saving the results')
    #print('The centeroid is: %5d' % model.center)

    optimizer.zero_grad()
    # loss.backward()
    optimizer.step()



    # if animatingTransformation:
    #     if i % 10 == 0 and i < 5000:
    #         # save optimized mesh per cycle
    #         model(1)[0].save_obj(os.path.join(args.output_dir, 'basilLeaf_%05d.obj' % i), save_texture=True)
    #
    # if i % 5000 == 0:
    #     # save optimized mesh
    #     model(1)[0].save_obj(os.path.join(args.output_dir, 'basilLeaf_%05d.obj' % i), save_texture=True)
    # # save optimized mesh
    # model(1)[0].save_obj(os.path.join(args.output_dir, 'basilLeaf_.obj'), save_texture=True)


if __name__ == '__main__':
    main()
