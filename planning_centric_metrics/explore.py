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

import numpy as np
import json
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import (load_gt, add_center_dist,
                                          filter_eval_boxes)
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.evaluate import DetectionEval
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .data import compile_data
from .models import compile_model
from .tools import (plot_box, SimpleLoss, synthetic_noise_trunk,
                    safe_forward, PKLEval, get_nusc_maps,
                    get_scene2samp)
from .planning_kl import make_rgba, render_observation


def viz_data(version, dataroot='/data/nuscenes',
             map_folder='/data/nuscenes/mini/',
             ego_only=False, t_spacing=0.25, bsz=16,
             num_workers=10, flip_aug=True):
    """Visualize the training/validation data.
    """
    trainloader, valloader = compile_data(version, dataroot, map_folder,
                                          ego_only, t_spacing,
                                          bsz, num_workers, flip_aug)
    def style_ax():
        plt.grid(b=None)
        plt.xticks([])
        plt.yticks([])
    for ix, (xs, ys) in enumerate(valloader):
        for rowi, (x, y) in enumerate(zip(xs, ys)):
            fig = plt.figure(figsize=(6*(x.shape[0]+2), 6))
            gs = mpl.gridspec.GridSpec(1, x.shape[0]+2)

            for i in range(x.shape[0]):
                plt.subplot(gs[0, i])
                plt.imshow(x[i].T, origin='lower', vmin=0, vmax=1)
                style_ax()

            plt.subplot(gs[0, i+1])
            for ti in range(y.shape[0]):
                first = y[ti].max(0)
                second = first.values.max(0)
                second_ix = second.indices.item()
                first_ix = first.indices[second_ix].item()
                plt.plot([first_ix], [second_ix], '.')
            plt.imshow(y.sum(0).T, origin='lower', vmin=0, vmax=1, alpha=0.0)
            style_ax()

            plt.subplot(gs[0, i+2])
            render_observation(x)

            plt.tight_layout()
            imname = f'check{ix:06}_{rowi:02}.jpg'
            print('saving', imname)
            plt.savefig(imname)
            plt.close(fig)


def scrape_masks(version, out_name='masks.json', dataroot='/data/nuscenes',
                 map_folder='/data/nuscenes/mini/',
                 ego_only=False, t_spacing=0.25, bsz=16,
                 num_workers=10, flip_aug=False):
    """Scrape a mask of where the car might exist at future timesteps.
    """
    trainloader, valloader = compile_data(version, dataroot, map_folder,
                                          ego_only, t_spacing, bsz,
                                          num_workers,
                                          flip_aug=flip_aug, only_y=True)

    masks = np.zeros((len(trainloader.dataset.local_ts),
                      trainloader.dataset.nx,
                      trainloader.dataset.ny))
    for ix, ys in enumerate(tqdm(trainloader)):
        update = (ys.sum(0) > 0).numpy()
        masks[update] = 1.0
        # symmetry
        masks[np.flip(update, 2)] = 1.0

    print('saving', out_name)
    with open(out_name, 'w') as writer:
        json.dump(masks.tolist(), writer)


def viz_masks(out_name, imname='masks.jpg'):
    """Visualize the masks.
    """
    with open(out_name, 'r') as reader:
        data = json.load(reader)

    fig = plt.figure(figsize=(16*4, 1*4))
    gs = mpl.gridspec.GridSpec(1, 16)

    for maski, mask in enumerate(data):
        plt.subplot(gs[0, maski])
        plt.imshow(np.array(mask).T, origin='lower', vmin=0, vmax=1)
        plt.title(f'N Occupied cells: {np.array(mask).sum()}')

    plt.tight_layout()
    print('saving', imname)
    plt.savefig(imname)
    plt.close(fig)


def eval_viz(version, modelpath,
             dataroot='/data/nuscenes', map_folder='/data/nuscenes/mini/',
             ego_only=True, t_spacing=0.5, bsz=8, num_workers=10,
             flip_aug=False, dropout_p=0.0, mask_json='masks_trainval.json',
             gpuid=0):
    """Visualize a planner's predictions.
    """
    device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
        else torch.device('cpu')
    print(f'using device: {device}')

    trainloader, valloader = compile_data(version, dataroot, map_folder,
                                          ego_only, t_spacing,
                                          bsz, num_workers, flip_aug)

    model = compile_model(cin=5, cout=16, with_skip=True,
                          dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    loss_fn = SimpleLoss(mask_json, 10.0, True, device)

    plt.figure(figsize=(4, 4))
    gs = mpl.gridspec.GridSpec(1, 1)

    with torch.no_grad():
        for batchi, (x, y) in enumerate(valloader):
            pred = model(x.to(device))
            pred = (pred.sigmoid().cpu() * 3.0).clamp(0, 1)
            pred[:, ~loss_fn.masks] = 0.0

            for ix in range(pred.shape[0]):

                plt.subplot(gs[0, 0])

                render_observation(x[ix])

                for maski, mask in enumerate(pred[ix]):
                    showimg = make_rgba(mask.numpy().T, (0.5, 0.0, 1.0))
                    plt.imshow(showimg, origin='lower')

                imname = f'val{batchi:06}_{ix:03}.jpg'
                plt.tight_layout()
                print('saving', imname)
                plt.savefig(imname)
                plt.clf()


def false_neg_viz(version, modelpath,
                  dataroot='/data/nuscenes', map_folder='/data/nuscenes/mini/',
                  ego_only=True, t_spacing=0.5, bsz=8,
                  num_workers=10, flip_aug=False,
                  dropout_p=0.0, mask_json='masks_trainval.json',
                  pos_weight=10.0, loss_clip=True, gpuid=0):
    """Remove each true detection to measure the "importance" of each object.
    """
    device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
        else torch.device('cpu')
    print(f'using device: {device}')

    trainloader, valloader = compile_data(version, dataroot, map_folder,
                                          ego_only, t_spacing,
                                          bsz, num_workers, flip_aug)

    model = compile_model(cin=5, cout=16, with_skip=True,
                          dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    loss_fn = SimpleLoss(mask_json, pos_weight, loss_clip, device)

    dataset = valloader.dataset

    with torch.no_grad():
        for batchi in range(len(dataset)):
            scene, name, t0 = dataset.ixes[batchi]
            lmap, centerlw, lobjs, lws, _ = dataset.get_state(scene, name, t0)
            x = dataset.render(lmap, centerlw, lobjs, lws)
            x = torch.Tensor(x).unsqueeze(0)
            pred = model(x.to(device))
            pred = torch.cat([pred for _ in range(len(lobjs))])
            pred_sig = pred.sigmoid()

            xs = []
            for drop_ix in range(len(lobjs)):
                new_objs = [obj for obji, obj in enumerate(lobjs)
                            if obji != drop_ix]
                new_lws = [lw for obji, lw in enumerate(lws)
                           if obji != drop_ix]

                x = dataset.render(lmap, centerlw, new_objs, new_lws)
                xs.append(torch.Tensor(x))
            xs = torch.stack(xs)
            preds = model(xs.to(device))

            preds = preds[:, loss_fn.masks]
            pred_sig = pred_sig[:, loss_fn.masks]
            pred = pred[:, loss_fn.masks]
            pkls = (F.binary_cross_entropy_with_logits(preds, pred_sig,
                                                       reduction='none')
                    - F.binary_cross_entropy_with_logits(pred, pred_sig,
                                                         reduction='none')
                    ).sum(1)

            fig = plt.figure(figsize=(3, 3))
            gs = mpl.gridspec.GridSpec(1, 1, left=0, bottom=0, right=1, top=1,
                                       wspace=0, hspace=0)

            ax = plt.subplot(gs[0, 0])
            # plot map
            for layer_name in dataset.layer_names:
                for poly in lmap[layer_name]:
                    plt.fill(poly[:, 0], poly[:, 1], 'g')
            for poly in lmap['road_divider']:
                plt.plot(poly[:, 0], poly[:, 1], c=(159./255., 0.0, 1.0))
            for poly in lmap['lane_divider']:
                plt.plot(poly[:, 0], poly[:, 1], c=(0.0, 0.0, 1.0))

            # plot objects
            for lobj, lw, pkl in zip(lobjs, lws, pkls):
                plot_box(lobj, lw, 'r',
                         alpha=(pkl / 12.0).clamp(0.1, 1).item())

            # plot ego
            plot_box([0.0, 0.0, 1.0, 0.0], [4.084, 1.73], 'b')

            plt.xlim((-17, 60))
            plt.ylim((-38.5, 38.5))
            ax.set_aspect('equal')
            plt.axis('off')

            imname = f'fneg{batchi:06}.jpg'
            print('saving', imname)
            plt.savefig(imname)
            plt.close(fig)


def false_pos_viz(version, modelpath,
                  dataroot='/data/nuscenes', map_folder='/data/nuscenes/mini/',
                  ego_only=True, t_spacing=0.5, bsz=8,
                  num_workers=10, flip_aug=False,
                  dropout_p=0.0, mask_json='masks_trainval.json',
                  pos_weight=10.0, loss_clip=True, gpuid=0):
    """Add a false positive at each (x,y) position in a grid about the ego.
    """
    device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
        else torch.device('cpu')
    print(f'using device: {device}')

    trainloader, valloader = compile_data(version, dataroot, map_folder,
                                          ego_only, t_spacing,
                                          bsz, num_workers, flip_aug)

    model = compile_model(cin=5, cout=16, with_skip=True,
                          dropout_p=dropout_p).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    loss_fn = SimpleLoss(mask_json, pos_weight, loss_clip, device)

    plt.figure(figsize=(4, 4))
    gs = mpl.gridspec.GridSpec(1, 1)

    dataset = valloader.dataset

    plt.figure(figsize=(3, 3))
    gs = mpl.gridspec.GridSpec(1, 1, left=0, bottom=0, right=1,
                               top=1, wspace=0, hspace=0)

    nx = 32
    ny = 32
    dxi = 8
    dxj = 4
    with torch.no_grad():
        for batchi in range(len(dataset)):
            scene, name, t0 = dataset.ixes[batchi]
            lmap, centerlw, lobjs, lws, _ = dataset.get_state(scene, name, t0)
            x = dataset.render(lmap, centerlw, lobjs, lws)
            x = torch.Tensor(x).unsqueeze(0)
            pred = model(x.to(device))
            pred_sig = pred.sigmoid()

            xixes = torch.linspace(0, x.shape[2]-1, nx).long()
            yixes = torch.linspace(0, x.shape[3]-1, ny).long()
            xs = []
            for i in xixes:
                for j in yixes:
                    imgij = x.clone()
                    loweri = max(0, j-dxi)
                    upperi = j+dxi
                    lowerj = max(0, x.shape[2] - i-dxj)
                    upperj = x.shape[2] - i+dxj
                    imgij[0, 3,  loweri:upperi, lowerj:upperj] = 1
                    xs.append(imgij)
            xs = torch.cat(xs)

            pkls = safe_forward(model, xs, device, pred, pred_sig, loss_fn)
            pkls = pkls.view(1, 1, nx, ny)
            up_pkls = torch.nn.Upsample(size=(x.shape[2], x.shape[3]),
                                        mode='bilinear',
                                        align_corners=True)(pkls)
            up_pkls = (up_pkls / 150.0).clamp(0.05, 1.0)

            plt.subplot(gs[0, 0])
            render_observation(x.squeeze(0))
            showimg = make_rgba(up_pkls[0, 0], (0, 0, 0.2))
            plt.imshow(np.flip(showimg, 0), origin='lower')

            imname = f'fpos{batchi:06}.jpg'
            print('saving', imname)
            plt.savefig(imname)
            plt.clf()


def eval_test(version, eval_set, result_path, modelpath='./planner.pt',
              dataroot='/data/nuscenes',
              map_folder='/data/nuscenes/mini/',
              nworkers=10, plot_kextremes=5,
              gpuid=0, mask_json='./masks_trainval.json',
              save_output=None):
    """Evaluate detections with PKL.
    """
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=True)
    nusc_maps = get_nusc_maps(map_folder)
    cfg = config_factory('detection_cvpr_2019')
    device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
        else torch.device('cpu')
    print(f'using device: {device}')

    nusc_eval = PKLEval(nusc, config=cfg, result_path=result_path,
                        eval_set=eval_set,
                        output_dir='./res', verbose=True)
    info = nusc_eval.pkl(nusc_maps, device, nworkers=nworkers,
                         plot_kextremes=plot_kextremes,
                         modelpath=modelpath,
                         mask_json=mask_json)
    print({k: v for k,v in info.items() if k != 'full'})
    if save_output is not None:
        print('saving results to', save_output)
        with open(save_output, 'w') as writer:
            json.dump(info, writer)


def generate_perfect(version, eval_set, dataroot='/data/nuscenes'):
    """Generate perfect detections.
    """
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=True)
    cfg = config_factory('detection_cvpr_2019')

    gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=True)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=True)

    submission = synthetic_noise_trunk(gt_boxes)

    outname = f'perfect_{version}_{eval_set}.json'
    print('saving', outname)
    with open(outname, 'w') as writer:
        json.dump(submission, writer)


def generate_drop_noise(version, eval_set, drop_p, dataroot='/data/nuscenes'):
    """Generate pseudo submissions where every box is dropped with
    probability p.
    """
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=True)
    cfg = config_factory('detection_cvpr_2019')

    gt_boxes = load_gt(nusc, eval_set, DetectionBox, verbose=True)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=True)

    for drop_p in [drop_p]:
        submission = synthetic_noise_trunk(gt_boxes, drop_p=drop_p)
        outname = f'perfect_{version}_{eval_set}_{drop_p}.json'
        print('saving', outname)
        with open(outname, 'w') as writer:
            json.dump(submission, writer)


def og_detection_eval(version, eval_set, result_path,
                      dataroot='/data/nuscenes'):
    """Evaluate according to NDS.
    """
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=True)
    cfg = config_factory('detection_cvpr_2019')
    nusc_eval = DetectionEval(nusc, config=cfg, result_path=result_path,
                              eval_set=eval_set,
                              output_dir='./res', verbose=True)
    nusc_eval.main(plot_examples=0, render_curves=False)


def pkl_distribution_plot(version, pkl_result_path, plot_name='dist.jpg',
                          ntrials=10, dataroot='/data/nuscenes'):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=True)

    with open(pkl_result_path, 'r') as reader:
        info = json.load(reader)
    vals = [val for val in info['full'].values()]

    # need to get scene->samples (ordered by timestamp)
    scene2samp = get_scene2samp(nusc, info['full'])

    fig = plt.figure(figsize=(15, 4))
    gs = mpl.gridspec.GridSpec(1, 3)

    ax = plt.subplot(gs[0, 0])
    plt.hist(vals, bins=40, range=(0, 2*info['mean']), label='Samples', color='b')
    plt.axvline(info['mean'], label=f"Overall Mean PKL: {info['mean']:.01f}", c='g', linestyle='dashed')
    plt.axvline(info['median'], label=f"Overall Median PKL: {info['median']:.01f}", c='r', linestyle='dashed')
    plt.xlabel('PKL')
    plt.ylabel('N samples')
    plt.legend()
    plt.xlim((0, 2*info['mean']))

    ax_mean = plt.subplot(gs[0, 1])
    for tr in range(ntrials):
        # random order
        vals = np.random.permutation(vals)
        xs = np.arange(1, len(vals)+ 1)
        # mean
        ys = [np.mean(vals[:i]) for i in xs]
        plt.sca(ax_mean)
        if tr == 0:
            plt.plot(xs, ys, 'b', alpha=0.5, label='Moving endpoint mean')
            plt.axhline(info['mean'], label=f"Overall Mean PKL: {info['mean']:.01f}", c='g', linestyle='dashed')
            plt.legend()
            plt.xlabel('N samples')
        else:
            plt.plot(xs, ys, 'b', alpha=0.5)
        # median
        ys = [np.median(vals[:i]) for i in xs]
        # plt.sca(ax_median)
        if tr == 0:
            plt.plot(xs, ys, 'r', alpha=0.5, label='Moving endpoint median')
            plt.axhline(info['median'], label=f"Overall Median PKL: {info['median']:.01f}", c='k', linestyle='dashed')
            plt.ylabel('PKL')
            plt.legend()
            plt.xlabel('N samples')
        else:
            plt.plot(xs, ys, 'r', alpha=0.5)
    # plt.axhline(info['median'], label=f"Overall Median PKL: {info['median']:.01f}", c='r', linestyle='dashed')
    ax_mean.set_ylim((0, 2*info['mean']))
    # ax_median.set_ylim((0, 2*info['median']))

    ax = plt.subplot(gs[0, 2])
    medians = []
    means = []
    for nperscene in range(1, 25):
        tokens = []
        for scene in scene2samp:
            ixes = np.round(np.linspace(0, len(scene2samp[scene]) - 1, nperscene))
            tokens.extend([scene2samp[scene][int(ix)] for ix in ixes])
        medians.append(
            (nperscene, np.median([info['full'][token] for token in tokens]))
            )
        means.append(
            (nperscene, np.mean([info['full'][token] for token in tokens]))
            )
    plt.plot([row[0] for row in means], [row[1] for row in means], 'b', label='Mean PKL')
    plt.plot([row[0] for row in medians], [row[1] for row in medians], 'r', label='Median PKL')
    plt.axhline(info['median'], label=f"Overall Median PKL: {info['median']:.01f}", c='r', linestyle='dashed')
    plt.axhline(info['mean'], label=f"Overall Mean PKL: {info['mean']:.01f}", c='b', linestyle='dashed')
    plt.legend()
    plt.xlabel('N samples per scene')
    plt.ylabel('PKL')

    plt.tight_layout()
    print('saving', plot_name)
    plt.savefig(plot_name)
    plt.close(fig)

