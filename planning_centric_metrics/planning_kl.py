# Copyright 2020 NVIDIA CORPORATION, Jonah Philion, Amlan Kar, Sanja Fidler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from .models import compile_model


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def objects2frame(history, center, toworld=False):
    """A sphagetti function that converts from global
    coordinates to "center" coordinates or the inverse.
    It has no for loops but works on batchs.
    """
    N, A, B = history.shape
    theta = np.arctan2(center[3], center[2])
    if not toworld:
        newloc = history[:, :, :2] - center[:2].reshape((1, 1, 2))
        rot = get_rot(theta).T
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) - theta
        newloc = np.dot(newloc.reshape((N*A, 2)), rot).reshape((N, A, 2))
    else:
        rot = get_rot(theta)
        newh = np.arctan2(history[:, :, 3], history[:, :, 2]) + theta
        newloc = np.dot(history[:, :, :2].reshape((N*A, 2)),
                        rot).reshape((N, A, 2))
    newh = np.stack((np.cos(newh), np.sin(newh)), 2)
    if toworld:
        newloc += center[:2]
    return np.append(newloc, newh, axis=2)


def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def raster_render(lmap, centerlw, lobjs, lws,
                  nx, ny, layer_names, line_names,
                  bx, dx):
    # draw both road layers vin one channel
    road_img = np.zeros((nx, ny))
    for layer_name in layer_names:
        for poly in lmap[layer_name]:
            # draw the lines
            pts = np.round(
                (poly - bx[:2] + dx[:2]/2.) / dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(road_img, [pts], 1.0)

    def draw_lane(layer_name, img):
        for poly in lmap[layer_name]:
            pts = np.round(
                (poly - bx[:2] + dx[:2]/2.) / dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.polylines(img, [pts], isClosed=False, color=1.0)
        return img
    road_div_img = np.zeros((nx, ny))
    draw_lane('road_divider', road_div_img)
    lane_div_img = np.zeros((nx, ny))
    draw_lane('lane_divider', lane_div_img)

    obj_img = np.zeros((nx, ny))
    for box, lw in zip(lobjs, lws):
        pts = get_corners(box, lw)
        # draw the box
        pts = np.round(
            (pts - bx[:2] + dx[:2]/2.) / dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(obj_img, [pts], 1.0)

    center_img = np.zeros((nx, ny))
    pts = get_corners([0.0, 0.0, 1.0, 0.0], centerlw)
    pts = np.round(
        (pts - bx[:2] + dx[:2]/2.) / dx[:2]
    ).astype(np.int32)
    pts[:, [1, 0]] = pts[:, [0, 1]]
    cv2.fillPoly(center_img, [pts], 1.0)

    return np.stack([road_img, road_div_img, lane_div_img,
                     obj_img, center_img])


def get_grid(point_cloud_range, voxel_size):
    lower = np.array(point_cloud_range[:(len(point_cloud_range) // 2)])
    upper = np.array(point_cloud_range[(len(point_cloud_range) // 2):])

    dx = np.array(voxel_size)
    bx = lower + dx/2.0
    nx = ((upper - lower) / dx).astype(int)

    return dx, bx, nx


def make_rgba(probs, color):
    H, W = probs.shape
    return np.stack((
                     np.full((H, W), color[0]),
                     np.full((H, W), color[1]),
                     np.full((H, W), color[2]),
                     probs,
                     ), 2)


def render_observation(x):
    # road
    showimg = make_rgba(x[0].numpy().T, (1.00, 0.50, 0.31))
    plt.imshow(showimg, origin='lower')

    # road div
    showimg = make_rgba(x[1].numpy().T, (159./255., 0.0, 1.0))
    plt.imshow(showimg, origin='lower')

    # lane div
    showimg = make_rgba(x[2].numpy().T, (0.0, 0.0, 1.0))
    plt.imshow(showimg, origin='lower')

    # objects
    showimg = make_rgba(x[3].numpy().T, (0.0, 0.0, 0.0))
    plt.imshow(showimg, origin='lower')

    # ego
    showimg = make_rgba(x[4].numpy().T, (0.0, 0.5, 0.0))
    plt.imshow(showimg, origin='lower')
    plt.grid(b=None)
    plt.xticks([])
    plt.yticks([])


def plot_heatmap(heat, masks):
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
              '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
              '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080', '#ffffff', '#000000']
    colors = [tuple(int(h[i:i+2], 16) for i in (1, 3, 5)) for h in colors]
    colors = [(c[0]/255, c[1]/255, c[2]/255) for c in colors]
    plot_heat = heat.clone()
    plot_heat[~masks] = 0
    for ti in range(plot_heat.shape[0]):
        flat = plot_heat[ti].view(-1)
        ixes = flat.topk(20).indices
        flat[ixes] = 1
        flat = flat.view(plot_heat.shape[1], plot_heat.shape[2])
        showimg = make_rgba(np.clip(flat.numpy().T, 0, 1), colors[ti])
        plt.imshow(showimg, origin='lower')


def analyze_plot(gtxs, predxs, gtdist_sig, preddist_sig, masks, pkls=None):
    for i, (gtx, predx, gtsig, predsig) in enumerate(zip(gtxs, predxs,
                                                         gtdist_sig,
                                                         preddist_sig)):
        fig = plt.figure(figsize=(9, 6))
        gs = mpl.gridspec.GridSpec(2, 3, left=0.01, bottom=0.01, right=0.99, top=0.99,
                                   wspace=0, hspace=0)
        ax = plt.subplot(gs[0, 0])
        render_observation(gtx)
        ax.annotate("Ground Truth", xy=(0.05, 0.95), xycoords="axes fraction")
        ax = plt.subplot(gs[0, 1])
        render_observation(predx)
        ax.annotate("Detections", xy=(0.05, 0.95), xycoords="axes fraction")

        ax = plt.subplot(gs[0, 2])
        new_obs = gtx.clone()
        new_obs[3] = 0
        render_observation(new_obs)
        showimg = make_rgba(np.clip((-gtx[3] + predx[3]).numpy().T, 0, 1),
                            (1.0, 0.0, 0.0))
        plt.imshow(showimg, origin='lower')
        showimg = make_rgba(np.clip((gtx[3] - predx[3]).numpy().T, 0, 1),
                            (1.0, 0.0, 1.0))
        plt.imshow(showimg, origin='lower')
        plt.legend(handles=[
            mpatches.Patch(color=(1.0, 0.0, 0.0), label='False Positive'),
            mpatches.Patch(color=(1.0, 0.0, 1.0), label='False Negative'),
        ], loc='upper right')
        if pkls is not None:
            ax.annotate(f"PKL: {pkls[i]:.2f}", xy=(0.05, 0.95),
                        xycoords="axes fraction")

        ax = plt.subplot(gs[1, 0])
        plot_heatmap(gtsig, masks)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(gs[1, 1])
        plot_heatmap(predsig, masks)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        imname = f'worst{i:04}.jpg'
        print('saving', imname)
        plt.savefig(imname)
        plt.close(fig)


def samp2ego(samp, nusc):
    egopose = nusc.get('ego_pose', nusc.get('sample_data',
                                            samp['data']['LIDAR_TOP'])
                       ['ego_pose_token'])
    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    return {
                'x': egopose['translation'][0],
                'y': egopose['translation'][1],
                'hcos': np.cos(rot),
                'hsin': np.sin(rot),
                'l': 4.084,
                'w': 1.73,
            }


def samp2mapname(samp, nusc):
    scene = nusc.get('scene', samp['scene_token'])
    log = nusc.get('log', scene['log_token'])
    return log['location']


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


def get_other_objs(boxes, ego):
    objs = []
    lws = []
    for box in boxes:
        rot = Quaternion(box.rotation).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])

        lws.append([box.size[1], box.size[0]])
        objs.append([box.translation[0], box.translation[1],
                     np.cos(rot), np.sin(rot)])
    objs = np.array(objs)
    lws = np.array(lws)
    if len(objs) == 0:
        lobjs = np.zeros((0, 4))
    else:
        lobjs = objects2frame(objs[np.newaxis, :, :],
                              ego,
                              )[0]
    return lobjs, lws


def collect_x(sample_token, nusc, gt_boxes, pred_boxes, nusc_maps, stretch,
              layer_names, line_names, nx, ny, bx, dx):
    samp = nusc.get('sample', sample_token)

    # ego location
    ego = samp2ego(samp, nusc)

    # local map
    map_name = samp2mapname(samp, nusc)
    lmap = get_local_map(nusc_maps[map_name],
                         [ego['x'], ego['y'], ego['hcos'], ego['hsin']],
                         stretch, layer_names, line_names)

    # detections
    gtlobjs, gtlws = get_other_objs(gt_boxes[sample_token],
                                    np.array([ego['x'], ego['y'],
                                              ego['hcos'], ego['hsin']]))
    predlobjs, predlws = get_other_objs(pred_boxes[sample_token],
                                        np.array([ego['x'], ego['y'],
                                                  ego['hcos'], ego['hsin']]))

    # render
    gtx = raster_render(lmap, [ego['l'], ego['w']], gtlobjs, gtlws,
                        nx, ny, layer_names, line_names,
                        bx, dx)
    predx = raster_render(lmap, [ego['l'], ego['w']], predlobjs, predlws,
                          nx, ny, layer_names, line_names,
                          bx, dx)

    return torch.Tensor(gtx), torch.Tensor(predx)


class EvalLoader(torch.utils.data.Dataset):
    def __init__(self, gt_boxes, pred_boxes, sample_tokens, nusc,
                 nusc_maps, stretch,
                 layer_names, line_names):
        self.dx, self.bx, (self.nx, self.ny) = get_grid([-17.0, -38.5, 60.0,
                                                         38.5], [0.3, 0.3])
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes
        self.sample_tokens = sample_tokens
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.stretch = stretch
        self.layer_names = layer_names
        self.line_names = line_names

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, index):
        return collect_x(self.sample_tokens[index], self.nusc, self.gt_boxes,
                         self.pred_boxes, self.nusc_maps, self.stretch,
                         self.layer_names, self.line_names,
                         self.nx, self.ny, self.bx, self.dx)


def calculate_pkl(gt_boxes, pred_boxes, sample_tokens, nusc,
                  nusc_maps, device, nworkers,
                  bsz=128, plot_kextremes=0, verbose=True,
                  modelpath='./planner.pt',
                  mask_json='./masks_trainval.json'):
    r""" Computes the PKL https://arxiv.org/abs/2004.08745. It is designed to
    consume boxes in the format from
    nuscenes.eval.detection.evaluate.DetectionEval.
    Args:
            gt_boxes (EvalBoxes): Ground truth objects
            pred_boxes (EvalBoxes): Predicted objects
            sample_tokens List[str]: timestamps to be evaluated
            nusc (NuScenes): parser object provided by nuscenes-devkit
            nusc_maps (dict): maps map names to NuScenesMap objects
            device (torch.device): device for running forward pass
            nworkers (int): number of workers for dataloader
            bsz (int): batch size for dataloader
            plot_kextremes (int): number of examples to plot
            verbose (bool): print or not
            modelpath (str): File path to model weights.
                             Will download if not found.
            mask_json (str): File path to trajectory masks.
                             Will download if not found.
    Returns:
            info (dict) : dictionary of PKL scores
    """

    # constants related to how the planner was trained
    layer_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    stretch = 70.0

    # load planner
    model = compile_model(cin=5, cout=16, with_skip=True,
                          dropout_p=0.0).to(device)
    if not os.path.isfile(modelpath):
        print(f'downloading model weights to location {modelpath}...')
        cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {modelpath}"
        print(f'running {cmd}')
        os.system(cmd)
    if verbose:
        print(f'using model weights {modelpath}')
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # load masks
    if not os.path.isfile(mask_json):
        print(f'downloading model masks to location {mask_json}...')
        cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=13M1xj9MkGo583ok9z8EkjQKSV8I2nWWF' -O {mask_json}"
        print(f'running {cmd}')
        os.system(cmd)
    if verbose:
        print(f'using location masks {mask_json}')
    with open(mask_json, 'r') as reader:
        masks = (torch.Tensor(json.load(reader)) == 1).to(device)

    dataset = EvalLoader(gt_boxes, pred_boxes, sample_tokens, nusc,
                         nusc_maps, stretch,
                         layer_names, line_names)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)

    if verbose:
        print('calculating pkl...')

    all_pkls = []
    for gtxs, predxs in tqdm(dataloader):
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))

        pkls = (F.binary_cross_entropy_with_logits(preddist[:, masks],
                                                   gtdist_sig[:, masks],
                                                   reduction='none')
                - F.binary_cross_entropy_with_logits(gtdist[:, masks],
                                                     gtdist_sig[:, masks],
                                                     reduction='none')).sum(1)

        all_pkls.append(pkls.cpu())
    all_pkls = torch.cat(all_pkls)

    # plot k extremes
    if verbose:
        print(f'plotting {plot_kextremes} timestamps...')
    if plot_kextremes > 0:
        worst_ixes = all_pkls.topk(plot_kextremes).indices
        out = [dataset[i] for i in worst_ixes]
        gtxs, predxs = list(zip(*out))
        gtxs, predxs = torch.stack(gtxs), torch.stack(predxs)
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))
        analyze_plot(gtxs, predxs, gtdist_sig.cpu(), preddist.sigmoid().cpu(),
                     masks.cpu(), pkls=all_pkls[worst_ixes])

    info = {
        'min': all_pkls.min().item(),
        'max': all_pkls.max().item(),
        'mean': all_pkls.mean().item(),
        'median': all_pkls.median().item(),
        'std': all_pkls.std().item(),
        'full': {tok: pk.item() for tok,pk in zip(sample_tokens, all_pkls)},
    }

    return info
