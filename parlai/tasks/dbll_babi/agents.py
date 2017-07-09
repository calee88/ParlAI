# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build

import copy
import os

tasks = {}
tasks[1] = 'rl1_pure_imitation'
tasks[2] = 'rl2_pos_neg'
tasks[3] = 'rl3_with_ans'
tasks[4] = 'rl4_with_hints'
tasks[5] = 'rl5_told_sf'
tasks[6] = 'rl6_only_some_rewards'
tasks[7] = 'rl7_no_feedback'
tasks[8] = 'rl8_imitation_plus_rl'
tasks[9] = 'rl9_ask_for_answer'
tasks[10] = 'rl10_ask_for_sf'

_suffixes = {
    'train': 'train',
    'test': 'test',
    'valid': 'dev'
}


def _path(subdir, task, opt, dt=''):
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    task_name = '%s_%s' % (task.split('_')[1],
                           tasks[int(task.split('_')[0])])
    return os.path.join(opt['datapath'], 'DBLL', 'dbll',
                        '{subdir}_{task}_{suffix}.txt'.format(
                            subdir=subdir, task=task_name,
                            suffix=_suffixes[dt]))


class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        params = opt['task'].split(':')[2]
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(os.path.join('babi', 'babi1'), params, opt)
        opt['cands_datafile'] = _path(os.path.join('babi', 'babi1'), params,
                                      opt, 'train')
        super().__init__(opt, shared)


# Defaults to task 2 with p=0.5.
class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = '2_p0.5'
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(os.path.join('babi', 'babi1'), task, opt)
        opt['cands_datafile'] = _path(os.path.join('babi', 'babi1'), task,
                                      opt, 'train')
        super().__init__(opt, shared)
