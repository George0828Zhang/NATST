# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
# import numpy as np
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from typing import Optional
from fairseq import utils, metrics
from fairseq.criterions import (
    FairseqCriterion, 
    register_criterion,
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterionConfig,
    LabelSmoothedCrossEntropyCriterion
)
from fairseq.dataclass import ChoiceEnum
# from fairseq.data.data_utils import post_process
# from fairseq.logging.meters import safe_round
from fairseq.tasks import FairseqTask
import logging
from omegaconf import II

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    objectives: Optional[ChoiceEnum(["asr", "st", "mtl", "mt"])] = II("task.objectives") # can be "mtl", "asr" or "st".
    asr_factor: Optional[float] = field(
        default=0.5,
        metadata={"help": "if objectives is 'mtl', then loss is "
        "calculated as: factor*asr_loss + (1-factor)*st_loss "},
    )
    decoder_use_ctc: bool = field(
        default=False,
        metadata={"help": "use ctcloss for decoder loss."},
    )
    zero_infinity: Optional[bool] = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )

@register_criterion(
    "multi_task_criterion", dataclass=MultiTaskCriterionConfig
)
class MultiTaskCriterion(LabelSmoothedCrossEntropyCriterion):  
    def __init__(self, cfg: MultiTaskCriterionConfig, task: FairseqTask):
        super().__init__(
            task, 
            cfg.sentence_avg, 
            cfg.label_smoothing,
            ignore_prefix_size=cfg.ignore_prefix_size,
            report_accuracy=cfg.report_accuracy
        )
        self.objectives = cfg.objectives
        self.asr_factor = cfg.asr_factor
        self.decoder_use_ctc = cfg.decoder_use_ctc
        if self.decoder_use_ctc:
            logger.info("Using ctc loss for decoder!")
            
        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.zero_infinity = cfg.zero_infinity

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        if self.decoder_use_ctc:
            loss, nll_loss = self.compute_ctc_loss(model, net_output, sample["target"], reduce=reduce)
        else:
            # original label smoothed xentropy loss by fairseq
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        asr_loss = None
        if self.objectives == "mtl":
            extra = net_output[1]
            asr_loss, _ = self.compute_ctc_loss(
                model, 
                net_output=extra["asr_out"], 
                target=sample["asr_target"], 
                reduce=reduce
            )
            loss = (1.-self.asr_factor)*loss + self.asr_factor*asr_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "asr_loss": 0 if asr_loss is None else asr_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, net_output, target, reduce=True):
        """
        lprobs is expected to be batch first. (from model forward output, or net_output)
        """
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        )
        bsz = target.size(0)
        # reshape lprobs to (L,B,X) for torch.ctc
        if lprobs.size(0) != bsz:
            raise RuntimeError(
                f'batch size error: lprobs shape={lprobs.size()}, bsz={bsz}')
        max_src = lprobs.size(1)
        lprobs = lprobs.transpose(1,0).contiguous()
        
        # get subsampling padding mask & lengths
        if net_output[1]["padding_mask"] is not None:
            non_padding_mask = ~net_output[1]["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        pad_mask = (target != self.pad_idx) & (
            target != self.eos_idx
        )
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.long().sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            nll_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        # label smoothing
        smooth_loss = -lprobs.sum(dim=-1).transpose(1,0) # (L,B) -> (B,L)
        if net_output[1]["padding_mask"] is not None:
            smooth_loss.masked_fill_(
                net_output[1]["padding_mask"],
                0.0
            )        
        eps_i = self.eps / lprobs.size(-1)
        loss = (1.0 - self.eps) * nll_loss + eps_i * smooth_loss.sum()

        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)
        
        asr_loss_sum = sum(log.get("asr_loss", 0) for log in logging_outputs)
        # ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "asr_loss", asr_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
