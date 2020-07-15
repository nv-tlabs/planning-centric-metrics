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
from tensorboardX import SummaryWriter
import os

from .data import compile_data
from .models import compile_model
from .tools import SimpleLoss, eval_model


def train(version, dataroot='/data/nuscenes',
          map_folder='/data/nuscenes/mini/',
          ego_only=True, t_spacing=0.25, bsz=16, num_workers=10, flip_aug=True,
          pos_weight=10.0, loss_clip=True, lr=2e-3, weight_decay=1e-5,
          dropout_p=0.0, nepochs=10000, mask_json='masks_trainval.json',
          logdir='./runs', gpuid=0):
    device = torch.device(f'cuda:{gpuid}') if gpuid >= 0\
        else torch.device('cpu')
    print(f'using device: {device}')

    trainloader, valloader = compile_data(version, dataroot, map_folder,
                                          ego_only, t_spacing,
                                          bsz, num_workers, flip_aug)

    model = compile_model(cin=5, cout=16, with_skip=True,
                          dropout_p=dropout_p).to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    loss_fn = SimpleLoss(mask_json, pos_weight, loss_clip, device)

    writer = SummaryWriter(logdir)
    counter = 0

    for epoch in range(nepochs):
        for batchi, (x, y) in enumerate(trainloader):
            opt.zero_grad()
            pred = model(x.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            opt.step()
            counter += 1

            # validation
            if counter % 2000 == 0:
                acc = eval_model(valloader, model, loss_fn, device)
                for t, ac in enumerate(acc['top5']):
                    writer.add_scalar(f'eval/acc{t}', ac, counter)
                writer.add_scalar('eval/time', acc['time'], counter)
                writer.add_scalar('eval/avg', acc['top5'].mean(), counter)

            # training
            if counter % 10 == 0:
                print(epoch, batchi, counter, loss.detach().item())
                writer.add_scalar('train/loss', loss.item(), counter)
                writer.add_scalar('train/epoch', epoch, counter)

            # checkpoint
            if counter % 2000 == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
