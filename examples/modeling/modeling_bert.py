import logging
import math
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (BertPreTrainedModel, BertModel)
from .utils_modeling import (BiAAttention, BiLinear)

logger = logging.getLogger(__name__)

class BertForDependencyParsing(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForDependencyParsing, self).__init__(config)
        self.use_postag = config.use_postag
        if self.use_postag:
            # 采用加的方式整合postag embedding
            # 使用0作为pad的postag位置
            self.postag_embeddings = nn.Embedding(config.num_postags, config.hidden_size, padding_idx=0)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.arc_h = nn.Linear(config.hidden_size, config.arc_space)
        self.arc_c = nn.Linear(config.hidden_size, config.arc_space)
        self.attention = BiAAttention(config.arc_space, config.arc_space, 1, biaffine=True)

        self.label_h = nn.Linear(config.hidden_size, config.label_space)
        self.label_c = nn.Linear(config.hidden_size, config.label_space)
        self.bilinear = BiLinear(config.label_space, config.label_space, self.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                postag_ids=None, head_ids=None, label_ids=None):
        # 1. 编码
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(sequence_output))
        arc_c = F.elu(self.arc_c(sequence_output))

        # output size [batch, length, label_space]
        label_h = F.elu(self.label_h(sequence_output))
        label_c = F.elu(self.label_c(sequence_output))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arcs = torch.cat([arc_h, arc_c], dim=1)
        labels = torch.cat([label_h, label_c], dim=1)

        arcs = self.dropout(arcs)
        arc_h, arc_c = arcs.chunk(2, 1)

        labels = self.dropout(labels)
        label_h, label_c = labels.chunk(2, 1)
        label_h = label_h.contiguous()
        label_c = label_c.contiguous()

        # [batch, length, length]
        out_arc = self.attention(arc_h, arc_c, mask_d=attention_mask, mask_e=attention_mask).squeeze(dim=1)

        batch, max_len, label_space = label_h.size()

        if head_ids is not None and label_ids is not None:
            # create batch index [batch]
            batch_index = torch.arange(0, batch).type_as(out_arc).long()
            # get vector for head_ids [batch, length, label_space],
            label_h = label_h[batch_index, head_ids.t()].transpose(0, 1).contiguous()
            # compute output for type [batch, length, num_labels]
            out_label = self.bilinear(label_h, label_c)

            # mask invalid position to -inf for log_softmax
            if attention_mask is not None:
                minus_inf = -1e8
                minus_mask = (1 - attention_mask) * minus_inf
                out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            # loss_arc shape [batch, length, length]
            loss_arc = F.log_softmax(out_arc, dim=1)
            # loss_label shape [batch, length, num_labels]
            loss_label = F.log_softmax(out_label, dim=2)

            # mask invalid position to 0 for sum loss
            if attention_mask is not None:
                loss_arc = loss_arc * attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
                loss_label = loss_label * attention_mask.unsqueeze(2)

                # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
                num = attention_mask.sum() - batch
            else:
                # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
                num = float(max_len - 1) * batch

            # first create index matrix [length, batch]
            child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
            child_index = child_index.type_as(out_arc).long()
            # [length-1, batch]
            loss_arc = loss_arc[batch_index, head_ids.t(), child_index][1:]
            loss_label = loss_label[batch_index, child_index, label_ids.t()][1:]

            total_loss = (-loss_arc.sum() / num) + (-loss_label.sum() / num)

            outputs = (total_loss, )

        else:
            label_h = label_h.unsqueeze(2).expand(batch, max_len, max_len, label_space).contiguous()
            label_c = label_c.unsqueeze(1).expand(batch, max_len, max_len, label_space).contiguous()

            # compute output for label [batch, length, length, num_labels]
            out_label = self.bilinear(label_h, label_c)

            # mask invalid position to -inf for log_softmax
            if attention_mask is not None:
                minus_inf = -1e8
                minus_mask = (1 - attention_mask) * minus_inf
                out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            # logits_arc shape [batch, length, length]
            logits_arc = F.log_softmax(out_arc, dim=1)
            # logits_label shape [batch, num_labels, length, length]
            logits_label = F.log_softmax(out_label, dim=3).permute(0, 3, 1, 2)

            # [batch, num_labels, length, length]
            energy = torch.exp(logits_arc.unsqueeze(1) + logits_label)

            outputs = (energy, logits_arc, logits_label, )

        return outputs


class BertForDependencyParsingWithOrder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForDependencyParsingWithOrder, self).__init__(config)
        self.use_postag = config.use_postag
        if self.use_postag:
            # 采用加的方式整合postag embedding
            # 使用0作为pad的postag位置
            self.postag_embeddings = nn.Embedding(config.num_postags, config.hidden_size, padding_idx=0)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.arc_h = nn.Linear(config.hidden_size, config.arc_space)
        self.arc_c = nn.Linear(config.hidden_size, config.arc_space)
        self.attention = BiAAttention(config.arc_space, config.arc_space, 1, biaffine=True)

        self.label_h = nn.Linear(config.hidden_size, config.label_space)
        self.label_c = nn.Linear(config.hidden_size, config.label_space)
        self.bilinear = BiLinear(config.label_space, config.label_space, self.num_labels)

        self.max_parsing_order = config.max_parsing_order
        self.order_hidden = nn.Linear(config.hidden_size, config.order_space)
        self.order_classifier = nn.Linear(config.order_space, self.max_parsing_order)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                postag_ids=None, order_ids=None, head_ids=None, label_ids=None):
        # 1. 编码
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(sequence_output))
        arc_c = F.elu(self.arc_c(sequence_output))

        # output size [batch, length, label_space]
        label_h = F.elu(self.label_h(sequence_output))
        label_c = F.elu(self.label_c(sequence_output))

        # output size [batch, length, order_space]
        order_h = F.elu(self.order_hidden(sequence_output))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arcs = torch.cat([arc_h, arc_c], dim=1)
        labels = torch.cat([label_h, label_c], dim=1)

        arcs = self.dropout(arcs)
        arc_h, arc_c = arcs.chunk(2, 1)

        labels = self.dropout(labels)
        label_h, label_c = labels.chunk(2, 1)
        label_h = label_h.contiguous()
        label_c = label_c.contiguous()

        order_h = self.dropout(order_h)

        # [batch, length, length]
        out_arc = self.attention(arc_h, arc_c, mask_d=attention_mask, mask_e=attention_mask).squeeze(dim=1)

        # [batch, length, max_parsing_order]
        logits_order = self.order_classifier(order_h)

        batch, max_len, label_space = label_h.size()

        if order_ids is not None and head_ids is not None and label_ids is not None:
            # create batch index [batch]
            batch_index = torch.arange(0, batch).type_as(out_arc).long()
            # get vector for head_ids [batch, length, label_space],
            label_h = label_h[batch_index, head_ids.t()].transpose(0, 1).contiguous()
            # compute output for type [batch, length, num_labels]
            out_label = self.bilinear(label_h, label_c)

            # mask invalid position to -inf for log_softmax
            if attention_mask is not None:
                minus_inf = -1e8
                minus_mask = (1 - attention_mask) * minus_inf
                out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            # loss_arc shape [batch, length, length]
            loss_arc = F.log_softmax(out_arc, dim=1)
            # loss_label shape [batch, length, num_labels]
            loss_label = F.log_softmax(out_label, dim=2)

            # mask invalid position to 0 for sum loss
            if attention_mask is not None:
                loss_arc = loss_arc * attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
                loss_label = loss_label * attention_mask.unsqueeze(2)

                # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
                num = attention_mask.sum() - batch
            else:
                # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
                num = float(max_len - 1) * batch

            # first create index matrix [length, batch]
            child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
            child_index = child_index.type_as(out_arc).long()
            # [length-1, batch]
            loss_arc = loss_arc[batch_index, head_ids.t(), child_index][1:]
            loss_label = loss_label[batch_index, child_index, label_ids.t()][1:]

            # 
            loss_fct = CrossEntropyLoss(reduction="sum")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_order = logits_order.view(-1, self.max_parsing_order)[active_loss]
                active_order_ids = order_ids.view(-1)[active_loss]
                loss_order = loss_fct(active_logits_order, active_order_ids)
            else:
                loss_order = loss_fct(logits_order.view(-1, self.max_parsing_order), order_ids.view(-1))

            total_loss = (-loss_arc.sum() / num) + (-loss_label.sum() / num) + (loss_order / num)

            outputs = (total_loss, )

        else:
            label_h = label_h.unsqueeze(2).expand(batch, max_len, max_len, label_space).contiguous()
            label_c = label_c.unsqueeze(1).expand(batch, max_len, max_len, label_space).contiguous()

            # compute output for label [batch, length, length, num_labels]
            out_label = self.bilinear(label_h, label_c)

            # mask invalid position to -inf for log_softmax
            if attention_mask is not None:
                minus_inf = -1e8
                minus_mask = (1 - attention_mask) * minus_inf
                out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

            # logits_arc shape [batch, length, length]
            logits_arc = F.log_softmax(out_arc, dim=1)
            # logits_label shape [batch, num_labels, length, length]
            logits_label = F.log_softmax(out_label, dim=3).permute(0, 3, 1, 2)

            # [batch, num_labels, length, length]
            energy = torch.exp(logits_arc.unsqueeze(1) + logits_label)

            outputs = (energy, logits_arc, logits_label, logits_order)

        return outputs
