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

from fire import Fire

from planning_centric_metrics import explore, train


if __name__ == '__main__':
    Fire({
        'viz_data': explore.viz_data,
        'scrape_masks': explore.scrape_masks,
        'viz_masks': explore.viz_masks,
        'eval_viz': explore.eval_viz,
        'false_neg_viz': explore.false_neg_viz,
        'false_pos_viz': explore.false_pos_viz,
        'eval_test': explore.eval_test,
        'generate_perfect': explore.generate_perfect,
        'og_detection_eval': explore.og_detection_eval,
        'generate_drop_noise': explore.generate_drop_noise,
        'pkl_distribution_plot': explore.pkl_distribution_plot,

        'train': train.train,
    })