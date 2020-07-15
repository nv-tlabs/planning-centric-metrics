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
from nuscenes.nuscenes import NuScenes
import os
import torch.utils.data as torchdata
from nuscenes.utils.splits import create_splits_scenes
from tqdm import tqdm
from pyquaternion import Quaternion
from scipy.interpolate import interp1d
import numpy as np
import cv2

from .tools import get_nusc_maps
from .planning_kl import get_grid, get_local_map, objects2frame, get_corners


class ClusterLoader(torchdata.Dataset):
    def __init__(self, nusc, nusc_maps, ego_only, flip_aug, is_train,
                 t_spacing, only_y):
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.is_train = is_train
        self.ego_only = ego_only
        self.t_spacing = t_spacing  # seconds
        self.stretch = 70.0  # meters
        self.only_y = only_y
        self.flip_aug = flip_aug
        self.local_ts = np.arange(0.25, 4.1, 0.25)  # seconds
        self.layer_names = ['road_segment', 'lane']
        self.line_names = ['road_divider', 'lane_divider']

        # rasterized ground is a nx x ny matrix.
        # real_coords = img_coords * dx + bx.
        self.dx, self.bx, (self.nx, self.ny) = get_grid([-17.0, -38.5,
                                                         60.0, 38.5],
                                                        [0.3, 0.3])

        self.scenes = self.get_scenes()
        self.scene2map = self.get_scene2map()
        self.data = self.compile_data()
        self.ixes = self.get_ixes()

        print(self)

    def __str__(self):
        return f"""Dataset loaded:
                   Nusc: {self.nusc.dataroot} | Nmaps: {len(self.nusc_maps)} |
                   Nscenes: {len(self.scenes)} | Length: {len(self)} |
                   ego_only: {self.ego_only} | t_spacing: {self.t_spacing} |
                   is_train: {self.is_train} | local_ts: {self.local_ts}"""

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]
        scenes = create_splits_scenes()[split]
        return scenes

    def get_scene2map(self):
        scene2map = {}
        for rec in self.nusc.scene:
            log = self.nusc.get('log', rec['log_token'])
            scene2map[rec['name']] = log['location']
        return scene2map

    def compile_data(self):
        recs = [rec for rec in self.nusc.sample]

        for rec in recs:
            rec['scene_name'] = self.nusc.get('scene',
                                              rec['scene_token'])['name']

        recs = [rec for rec in recs if rec['scene_name'] in self.scenes]

        recs = sorted(recs, key=lambda x: (x['scene_name'], x['timestamp']))

        scene2data = {}
        for rec in tqdm(recs):
            scene = rec['scene_name']
            if scene not in scene2data:
                scene2data[scene] = {}
                scene2data[scene]['ego'] = {'traj': [], 'w': 1.73,
                                            'l': 4.084, 'k': 'ego'}
            # add ego location
            egopose = self.nusc.get('ego_pose',
                                    self.nusc.get('sample_data',
                                                  rec['data']['LIDAR_TOP'])
                                    ['ego_pose_token'])
            rot = Quaternion(egopose['rotation']).rotation_matrix
            rot = np.arctan2(rot[1, 0], rot[0, 0])
            scene2data[scene]['ego']['traj'].append({
                'x': egopose['translation'][0],
                'y': egopose['translation'][1],
                'hcos': np.cos(rot),
                'hsin': np.sin(rot),
                't': egopose['timestamp'],
            })
            # add detection locations
            for ann in rec['anns']:
                instance = self.nusc.get('sample_annotation', ann)
                instance_name = instance['instance_token']
                rot = Quaternion(instance['rotation']).rotation_matrix
                rot = np.arctan2(rot[1, 0], rot[0, 0])
                if instance_name not in scene2data[scene]:
                    assert(instance_name != 'ego'), instance_name
                    scene2data[scene][instance_name] =\
                        {'traj': [], 'w': instance['size'][0],
                         'l': instance['size'][1],
                         'k': instance['category_name']}
                scene2data[scene][instance_name]['traj'].append({
                    'x': instance['translation'][0],
                    'y': instance['translation'][1],
                    'hcos': np.cos(rot),
                    'hsin': np.sin(rot),
                    't': rec['timestamp'],
                })

        return self.post_process(scene2data)

    def post_process(self, data):
        """Build the interpolaters so we can evaluate at any timestep.
        """
        scene2info = {}
        for scene in data:
            scene2info[scene] = {}
            for name in data[scene]:
                info = {}
                t = [row['t']*1e-6 for row in data[scene][name]['traj']]
                x = [[row['x'], row['y'], row['hcos'], row['hsin']]
                     for row in data[scene][name]['traj']]
                # make sure the object exists even if it only appears once
                if t[-1] == t[0]:
                    t.append(t[0] + 0.02)
                    x.append([val for val in x[-1]])
                # linearly interpolate
                # (using hcos and hsin is only roughly correct for heading)
                info['interp'] = interp1d(t, x, kind='linear', axis=0,
                                          copy=False, bounds_error=True,
                                          assume_sorted=True)
                info['lw'] = [data[scene][name]['l'], data[scene][name]['w']]
                info['k'] = data[scene][name]['k']
                info['tmin'] = t[0]
                info['tmax'] = t[-1]

                scene2info[scene][name] = info
        return scene2info

    def get_ixes(self):
        """Which vehicles should we predict and for what timesteps in which scenes.
        """
        ixes = []
        min_distance = 0.2  # meters
        for scene in self.data:
            if not self.ego_only:
                names = [name for name in self.data[scene]
                         if (self.data[scene][name]['k'].split('.')[0]
                             == 'vehicle'
                         and self.data[scene][name]['k'] != 'vehicle.trailer')
                         or self.data[scene][name]['k'] == 'ego'
                         ]
            else:
                names = ['ego']
            for name in names:
                ts = np.arange(self.data[scene][name]['tmin'],
                               self.data[scene][name]['tmax']
                               - self.local_ts[-1],
                               self.t_spacing)
                # remove cases where the car doesn't move
                for t in ts:
                    dist = np.linalg.norm(self.data[scene][name]['interp'](
                        t + self.local_ts[-1])[:2]
                        - self.data[scene][name]['interp'](t)[:2]
                        )
                    if name == 'ego' or dist > min_distance:
                        ixes.append((scene, name, t))

        return ixes

    def get_state(self, scene, name, t0):
        # ego object
        center = self.data[scene][name]['interp'](t0)
        centerlw = self.data[scene][name]['lw']

        # map
        lmap = get_local_map(self.nusc_maps[self.scene2map[scene]],
                             center, self.stretch, self.layer_names,
                             self.line_names)

        # other objects
        objs = np.array([row['interp'](t0)
                         for na, row in self.data[scene].items()
                         if na != name and row['tmin'] <= t0 <= row['tmax']])
        if len(objs) == 0:
            lobjs = np.zeros((0, 4))
        else:
            lobjs = objects2frame(objs[np.newaxis, :, :], center)[0]

        lws = np.array([row['lw'] for na, row in self.data[scene].items()
                        if na != name and row['tmin'] <= t0 <= row['tmax']])

        # tgt
        tgt = self.data[scene][name]['interp'](t0 + self.local_ts)
        ltgt = objects2frame(tgt[np.newaxis, :, :], center)[0]

        return lmap, centerlw, lobjs, lws, ltgt

    def render(self, lmap, centerlw, lobjs, lws):
        # draw both road layers vin one channel
        road_img = np.zeros((self.nx, self.ny))
        for layer_name in self.layer_names:
            for poly in lmap[layer_name]:
                # draw the lines
                pts = np.round(
                    (poly - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(road_img, [pts], 1.0)

        def draw_lane(layer_name, img):
            for poly in lmap[layer_name]:
                pts = np.round(
                    (poly - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.polylines(img, [pts], isClosed=False, color=1.0)
            return img
        road_div_img = np.zeros((self.nx, self.ny))
        draw_lane('road_divider', road_div_img)
        lane_div_img = np.zeros((self.nx, self.ny))
        draw_lane('lane_divider', lane_div_img)

        obj_img = np.zeros((self.nx, self.ny))
        for box, lw in zip(lobjs, lws):
            pts = get_corners(box, lw)
            # draw the box
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(obj_img, [pts], 1.0)

        center_img = np.zeros((self.nx, self.ny))
        pts = get_corners([0.0, 0.0, 1.0, 0.0], centerlw)
        pts = np.round(
            (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
        ).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]
        cv2.fillPoly(center_img, [pts], 1.0)

        return np.stack([road_img, road_div_img, lane_div_img,
                         obj_img, center_img])

    def get_tgt(self, ltgt):
        pts = np.round(
            (ltgt[:, :2] - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
        ).astype(np.int32)
        tgt = np.zeros((ltgt.shape[0], self.nx, self.ny))

        pts = np.concatenate((
                    np.arange(ltgt.shape[0], dtype=np.int32)[:, np.newaxis],
                    pts), 1)
        kept = np.logical_and(0 <= pts[:, 1], pts[:, 1] < self.nx)
        kept = np.logical_and(kept, 0 <= pts[:, 2])
        kept = np.logical_and(kept, pts[:, 2] < self.ny)

        pts = pts[kept]
        tgt[pts[:, 0], pts[:, 1], pts[:, 2]] = 1.0

        return tgt

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        """Map channels: road, road divider, lane divider
           Object channels: detected objects, ego object
        """
        scene, name, t0 = self.ixes[index]

        # option to only return y
        if self.only_y:
            center = self.data[scene][name]['interp'](t0)
            tgt = self.data[scene][name]['interp'](t0 + self.local_ts)
            ltgt = objects2frame(tgt[np.newaxis, :, :], center)[0]
            y = self.get_tgt(ltgt)
            if np.random.rand() > 0.5:
                y = np.flip(y, 2).copy()
            return torch.Tensor(y)

        lmap, centerlw, lobjs, lws, ltgt = self.get_state(scene, name, t0)

        x = self.render(lmap, centerlw, lobjs, lws)
        y = self.get_tgt(ltgt)
        if self.flip_aug:
            if np.random.rand() > 0.5:
                x = np.flip(x, 2).copy()
                y = np.flip(y, 2).copy()

        return torch.Tensor(x), torch.Tensor(y)


def worker_init_fn(x):
    np.random.seed(42 + x)


def compile_data(version, dataroot, map_folder, ego_only, t_spacing,
                 bsz, num_workers, flip_aug, only_y=False):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    nusc_maps = get_nusc_maps(map_folder)

    traindata = ClusterLoader(nusc, nusc_maps, ego_only, flip_aug=flip_aug,
                              is_train=True, t_spacing=t_spacing,
                              only_y=only_y)
    valdata = ClusterLoader(nusc, nusc_maps, ego_only=True, flip_aug=False,
                            is_train=False, t_spacing=0.5, only_y=only_y)
    trainloader = torchdata.DataLoader(traindata, batch_size=bsz, shuffle=True,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
    valloader = torchdata.DataLoader(valdata, batch_size=bsz, shuffle=False,
                                     num_workers=num_workers,
                                     worker_init_fn=worker_init_fn)

    return trainloader, valloader
