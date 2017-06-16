# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Script to run evaluation only on a pretrained model."""
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')
import logging

import pdb
import sys

from parlai.agents.drqa.utils import Timer
from parlai.agents.drqa.agents import DocReaderAgent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task


def main(opt, outputpath):
    # Check options
    assert('pretrained_model' in opt)
    assert(opt['datatype'] in {'test', 'valid'})

    # Calculate TDNN embedding dim (after applying TDNN to char tensor)
    opt['kernels'] = ''.join(opt['kernels'])
    if isinstance(opt['kernels'], str):
        opt['kernels'] = eval(opt['kernels']) # convert string list of tuple --> list of tuple

    if opt['add_char2word']:
        opt['embedding_dim_TDNN']=0
        for i, n in enumerate(opt['kernels']):
            opt['embedding_dim_TDNN'] += n[1]


    #Write prediction file
    #f_predict = open(("exp-squad/" + str(opt['expnum']) + '.prediction'),"w")
    f_predict = open(str(outputpath),"w")
    f_predict.write("{")

    # Load document reader
    doc_reader = DocReaderAgent(opt)

    valid_world = create_task(opt, doc_reader)
    valid_time = Timer()

    # Sent prediction
    valid_world.agents[1].opt['ans_sent_predict'] = False
    valid_world.agents[1].model.network.opt['ans_sent_predict'] = False  # disable sentence predicction by default
    if opt['ans_sent_predict']:
        valid_world.agents[1].model.input_idx_bdy -= 1

    nExample = 0
    f1_avg_prev = 0
    acc_avg_prev = 0
    for _ in valid_world:
        valid_world.parley()
        if nExample > 0:
            f_predict.write(", ")
        nExample+=1
        #pdb.set_trace()
        f_predict.write('"' + valid_world.acts[0]['reward'] + '": ')
        temp_valid_word = valid_world.acts[1]['text'].replace("\"", "\\\"")
        f_predict.write('"' + temp_valid_word + '"')

        f1_avg_cur = valid_world.agents[0].report()['f1']
        f1_cur = nExample*f1_avg_cur - (nExample-1)*f1_avg_prev
        f1_avg_prev = f1_avg_cur

    #pdb.set_trace()
    metrics = valid_world.report()

    # Close prediction file
    f_predict.write("}")
    f_predict.close()

if __name__ == '__main__':
    # Get command line arguments
    path = "evasquard" + sys.arg[1]
    defopt = "--pretrained_model exp_squard/exph13-fix-bi-ldecay -t squard --embedding_file /data3/calee/git/convai/ParlAI/data/glove.840B.300d.txt --dropout_rnn 0.3 --dropout_emb 0.3 --gpu 0 --qp_bottleneck True --qp_birnn True --lrate_decay True --model_file exp_squard/exph13-fix-bi-ldecay --datatype " + path
    argparser = ParlaiParser()
    DocReaderAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args(defopt)


    # Set cuda
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
         torch.cuda.set_device(opt['gpu'])

    # Run!
    main(opt, sys.arg[2])
