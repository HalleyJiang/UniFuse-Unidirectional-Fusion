# by Hualie Jiang (jainghualie0@gmail.com)

import os
from glob import glob
import cv2
import torch
import numpy as np
from PIL import Image

# define the unit sphere grid of a panorama
pano_w, pano_h = 2048, 1024
pu, pv = np.meshgrid(np.arange(pano_w, dtype=np.float32), np.arange(pano_h, dtype=np.float32))
theta = (pano_h-pv-0.5)/pano_h*np.pi
phi = 2*(pu-pano_w/2+0.5)/pano_w*np.pi

px = np.sin(theta)*np.sin(phi)
py = np.cos(theta)
pz = np.sin(theta)*np.cos(phi)


# define the warping grid from the panorama to the pinhole on the north pole
w, h = 768, 768
hfov, vfov = np.pi/2, np.pi/2
u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
x = (u-w/2+0.5)/(w/2)
y = (v-h/2+0.5)/(h/2)
z = np.ones_like(x)
r = np.sqrt(x**2+y**2+z**2)

nx = x/r
ny = -z/r
nz = y/r

n_theta = np.arccos(ny)
n_phi = np.arctan2(nx, nz)
nv = pano_h-n_theta*pano_h/np.pi-0.5
nu = n_phi*pano_w/np.pi/2+pano_w/2-0.5

pn_grid = torch.tensor(np.stack([2 * nu / pano_w - 1, 2 * nv / pano_h - 1], axis=2)).unsqueeze(0)

# define the warping grid from the pinhole on the north pole to the panorama
rx = px
ry = pz
rz = -py

x_n = rx/rz 
y_n = ry/rz

ru = x_n*(w/2)+w/2-0.5
rv = y_n*(h/2)+h/2-0.5
ru[rz<0.2] = -1000
rv[rz<0.2] = -1000

np_grid = torch.tensor(np.stack([2 * ru / w - 1, 2 * rv / h - 1], axis=2)).unsqueeze(0)


# define the warping grid from the panorama to the pinhole on the south pole
sx = x/r
sy = z/r
sz = -y/r
s_theta = np.arccos(sy)
s_phi = np.arctan2(sx, sz)

sv = pano_h-s_theta*pano_h/np.pi-0.5
su = s_phi*pano_w/np.pi/2+pano_w/2-0.5

ps_grid = torch.tensor(np.stack([2 * su / pano_w - 1, 2 * sv / pano_h - 1], axis=2)).unsqueeze(0)

# define the warping grid from the pinhole on the south pole to the panorama
rx = px
ry = -pz
rz = py
x_n = rx/rz 
y_n = ry/rz
ru = x_n*(w/2)+w/2-0.5
rv = y_n*(h/2)+h/2-0.5
ru[rz<0.2] = -1000
rv[rz<0.2] = -1000

sp_grid = torch.tensor(np.stack([2 * ru / w - 1, 2 * rv / h - 1], axis=2)).unsqueeze(0)

kernel = np.ones((15, 15), dtype=np.uint8)
scenes = sorted(glob('*/rgb/'))

n = 0
for path in scenes:
    files = glob(path+'*.png')
    print(path, len(files))
    n += len(files)
    os.makedirs(path.replace('rgb', 'rgb_'), exist_ok=True)
    for file in files:
        I = cv2.imread(file).astype(np.float32)
        img = torch.tensor(I).permute(2, 0, 1).unsqueeze(0)
        up = torch.nn.functional.grid_sample(img, pn_grid, align_corners=False)[0] \
            .permute(1, 2, 0).numpy().astype(np.uint8)

        mask = (up.sum(axis=2)==0).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, 3)
        r_up = cv2.inpaint(up, mask, 11, cv2.INPAINT_TELEA)
        r_up = torch.tensor(r_up.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

        p_up = torch.nn.functional.grid_sample(r_up, np_grid, align_corners=False)[0] \
            .permute(1, 2, 0).numpy()


        down = torch.nn.functional.grid_sample(img, ps_grid, align_corners=False)[0] \
            .permute(1, 2, 0).numpy()

        mask = (down.sum(axis=2)==0).astype(np.uint8)
        mask = cv2.dilate(mask, kernel, 3)
        r_down = cv2.inpaint(down.astype(np.uint8), mask, 11, cv2.INPAINT_TELEA)

        r_down = torch.tensor(r_down.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        p_down = torch.nn.functional.grid_sample(r_down, sp_grid, align_corners=False)[0] \
            .permute(1, 2, 0).numpy()

        p_ = p_up+p_down
        weight = 1/(1 + np.exp(-(I - p_))/10) 
        pano = weight*I+(1-weight)*p_
        pano[pano>255]=255
        pano[pano<0] = 0
        cv2.imwrite(file.replace('rgb/', 'rgb_/'), pano)

print(n)