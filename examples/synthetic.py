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
# limitations under the License.import os

from fire import Fire
import torch
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.config import config_factory
import os

from planning_centric_metrics import calculate_pkl


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                 map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def get_example_submission():
    cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Wl2j8E-rA-v2z7ghl3oSePGK2xeAMSrj' -O example_submission.json"
    print('downloading example submission...')
    os.system(cmd)


def quick_test(dataroot='/data/nuscenes',
               map_folder='/data/nuscenes/mini',
               gpuid=0, nworkers=10):
    """Evaluate detections with PKL.
    """
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=os.path.join(dataroot, 'mini'),
                    verbose=True)
    nusc_maps = get_nusc_maps(map_folder)
    cfg = config_factory('detection_cvpr_2019')
    device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
        else torch.device('cpu')
    print(f'using device: {device}')

    get_example_submission()

    nusc_eval = DetectionEval(nusc, config=cfg,
                              result_path='./example_submission.json',
                              eval_set='mini_train',
                              output_dir='./res', verbose=True)
    info = calculate_pkl(nusc_eval.gt_boxes, nusc_eval.pred_boxes,
                         nusc_eval.sample_tokens, nusc_eval.nusc,
                         nusc_maps, device,
                         nworkers, bsz=128,
                         plot_kextremes=5,
                         verbose=True)
    print(info)


if __name__ == '__main__':
    Fire({
        'quick_test': quick_test,
    })
