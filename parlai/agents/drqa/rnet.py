# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers

import pdb

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)

        # ...(maybe) keep them fixed
        if opt['fix_embeddings']:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # Register a buffer to (maybe) fill later for keeping *some* fixed
        if opt['tune_partial'] > 0:
            buffer_size = torch.Size((
                opt['vocab_size'] - opt['tune_partial'] - 2,
                opt['embedding_dim']
            ))
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])



        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % opt['merge_mode'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
            
        #if opt['question_merge'] == 'self_attn':
        # Q-P matching
        """
        birnn = False
        self.qp_match = layers.GatedAttentionRNN(
            input_size= question_hidden_size,
            hidden_size= opt['hidden_size'],
            padding=opt['rnn_padding'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            birnn=birnn,
            gate=True
        )
        qp_matched_size = opt['hidden_size']
        if birnn:
            qp_matched_size = qp_matched_size * 2
        """
        opt['qp_rnn_size'] = doc_hidden_size + question_hidden_size
        if opt['qp_bottleneck']:
            opt['qp_rnn_size'] = opt['hidden_size']
        
        self.qp_match = layers.GatedAttentionBilinearRNN(
            x_size = doc_hidden_size,
            y_size = question_hidden_size,            
            hidden_size= opt['qp_rnn_size'],
            padding=opt['rnn_padding'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            birnn=opt['qp_birnn'],
            concat = opt['qp_concat'],
            gate=True
        )
        qp_matched_size = opt['qp_rnn_size']
        if opt['qp_birnn']:
            qp_matched_size = qp_matched_size * 2
        if opt['qp_concat']:
            qp_matched_size = qp_matched_size + doc_hidden_size        
                    
                    
        ## PP matching: 
        opt['pp_rnn_size'] = qp_matched_size * 2
        if opt['pp_bottleneck']:
            opt['pp_rnn_size'] = opt['hidden_size']
        
        self.pp_match = layers.GatedAttentionBilinearRNN(
            x_size = qp_matched_size,
            y_size = qp_matched_size,            
            hidden_size= opt['pp_rnn_size'],
            padding=opt['rnn_padding'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            birnn=opt['pp_birnn'],
            concat = opt['pp_concat'],
            gate=False
        )
        pp_matched_size = opt['pp_rnn_size']
        if opt['pp_birnn']:
            pp_matched_size = pp_matched_size * 2
        if opt['pp_concat']:
            pp_matched_size = pp_matched_size + qp_matched_size
        
        
        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            pp_matched_size,
            question_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            pp_matched_size,
            question_hidden_size,
        )
        
        #pdb.set_trace()
        
    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input = torch.cat([x1_emb, x2_weighted_emb, x1_f], 2)
        else:
            drnn_input = torch.cat([x1_emb, x1_f], 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Encode question with RNN
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        
        # QP matching
        qp_matched_doc = self.qp_match(doc_hiddens, x1_mask, question_hiddens, x2_mask)
        
        # PP matching
        if not qp_matched_doc.is_contiguous():
            qp_matched_doc = qp_matched_doc.contiguous()
            
        pp_matched_doc = self.pp_match(qp_matched_doc, x1_mask, qp_matched_doc, x1_mask)
        print(pp_matched_doc.size())
        pdb.set_trace()
        
        # Merge question hiddens
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
       
        # Predict start and end positions
        start_scores = self.start_attn(pp_matched_doc, question_hidden, x1_mask)
        end_scores = self.end_attn(pp_matched_doc, question_hidden, x1_mask)
        return start_scores, end_scores




