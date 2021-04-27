# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Any
from fairseq import metrics, utils, scoring
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
# from fairseq.models import FairseqEncoderModel 

from fairseq.logging.meters import safe_round
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from omegaconf import II

from fairseq.scoring.bleu import SacrebleuConfig, SacrebleuScorer
from fairseq.scoring.wer import WerScorerConfig, WerScorer
from fairseq.scoring.tokenizer import EvaluationTokenizer

from .speech_to_text_multi_task_dataset import (
    SpeechToTextMultiTaskDataset,
    SpeechToTextMultiTaskDatasetCreator
)

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4

@dataclass
class SpeechToTextConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    max_source_positions: Optional[int] = field(
        default=6000, metadata={"help": "max number of tokens in the target sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    config_yaml: Optional[str] = field(
        default="config.yaml", metadata={"help": "Configuration YAML filename (under manifest root)"}
    )

    # train
    objectives: Optional[ChoiceEnum(["asr", "st", "mtl", "mt"])] = field(
        default="asr", metadata={"help": 'can be "asr,st", "asr", "mt" or "st"'}
    )

    # eval
    eval_bleu: Optional[bool] = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_print_samples: Optional[bool] = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    seed: int = II("common.seed")

    # generation
    post_process: Optional[str] = II("common_eval.post_process")
    generation: GenerationConfig = GenerationConfig()

    # evaluation
    eval_bleu_args: SacrebleuConfig = SacrebleuConfig()
    eval_wer_args: WerScorerConfig = WerScorerConfig()


@register_task("speech_to_text_multi_task", dataclass=SpeechToTextConfig)
class SpeechToTextMultiTask(SpeechToTextTask):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        # Handle tokenization
        # for speech_to_text, bpe & tokenizer configs are in S2TDataCfg, 
        # hence, passing None.
        self.tokenizer = self.build_tokenizer(None)
        self.objectives = args.objectives
    
    @staticmethod
    def add_args(parser):
        """ This is necessary because fairseq-generate does not work with omegaconf :( """
        SpeechToTextTask.add_args(parser)
        parser.add_argument(
            "--objectives",
            type=str,
            choices=["asr", "st", "mtl", "mt"],
            default="asr",
            help='can be "asr,st", "asr", "mt" or "st"',
        )
        

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextMultiTaskDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )
    
    @classmethod
    def build_dataset_for_inference(cls, audio_paths, n_frames):
        return SpeechToTextMultiTaskDataset("interactive", False, {}, audio_paths, n_frames)


    def build_model(self, args):
        model = super().build_model(args)
        if self.args.eval_bleu:
            self.sequence_generator = self.build_generator(
                [model], self.args.generation
            )
        return model

    def process_sample(self, sample):
        """
        Need to process here instead of criterion, because in inference time criterion is not called.

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "asr_prev_output_tokens": asr_prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "asr_target": asr_target,
            "asr_target_lengths": asr_target_lengths,
            "asr_ntokens": asr_ntokens,
            "nsentences": len(samples),
        }
        """
        # single objective
        if self.objectives == "asr":
            if "asr_prev_output_tokens" in sample["net_input"]:
                sample["net_input"]["prev_output_tokens"] = sample["net_input"]["asr_prev_output_tokens"]
            if "asr_target" in sample:
                sample["target"] = sample["asr_target"]
                sample["target_lengths"] = sample["asr_target_lengths"]
                sample["ntokens"] = sample["asr_ntokens"]
        elif self.objectives == "mt":
            sample["net_input"]["src_tokens"] = sample["asr_target"]
            sample["net_input"]["src_lengths"] = sample["asr_target_lengths"]

        if "asr_prev_output_tokens" in sample["net_input"]:
            del sample["net_input"]["asr_prev_output_tokens"]
        return sample

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        sample = self.process_sample(sample)
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        sample = self.process_sample(sample)
        with torch.no_grad():
            if getattr(models[0], "one_pass_decoding", False):
                # one-pass decoding
                if hasattr(self, 'blank_symbol'):
                    sample["net_input"]["blank_idx"] = self.tgt_dict.index(self.blank_symbol)
                return models[0].generate(**sample["net_input"])
            else:
                # incremental decoding
                return generator.generate(
                    models, sample, prefix_tokens=prefix_tokens, constraints=constraints
                )

    def valid_step(self, sample, model, criterion):
        sample = self.process_sample(sample)
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu, wer = self._inference_with_bleu_wer(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
            
            logging_output.update(wer)
        return loss, sample_size, logging_output

    def _inference_with_bleu_wer(self, generator, sample, model):
        # scorer = scoring.build_scorer(self.args.scoring, self.tgt_dict)
        import sacrebleu
        import editdistance

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.post_process,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer is not None:
                s = self.tokenizer.decode(s)
            return s if s else "UNKNOWNTOKENINHYP"

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(                
                decode(gen_out[i][0]["tokens"])
            )
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])

        bleu_scorer = SacrebleuScorer(self.args.eval_bleu_args)
        wer_scorer = WerScorer(self.args.eval_wer_args)

        for h, r in zip(hyps, refs):
            bleu_scorer.add_string(ref=r, pred=h)
            wer_scorer.add_string(ref=r, pred=h)

        bleu = bleu_scorer.sacrebleu.corpus_bleu(
            bleu_scorer.pred, [bleu_scorer.ref], tokenize="none"
        )
        wer = {
            "wv_errors": wer_scorer.distance, 
            "w_errors": wer_scorer.distance, 
            "w_total": wer_scorer.ref_length
        }

        return bleu, wer
        
        # bleu = sacrebleu.corpus_bleu(hyps, [refs])
           
        # w_errs = 0
        # w_len = 0
        # wv_errs = 0
        # for h, r in zip(hyps, refs):
        #     targ_words = h.split()
        #     pred_words = r.split()
        #     dist = editdistance.eval(pred_words, targ_words)
        #     w_errs += dist
        #     wv_errs += dist
        #     w_len += len(targ_words)

        # return bleu, {"wv_errors": wv_errs, "w_errors": w_errs, "w_total": w_len}

    # def begin_valid_epoch(self, epoch, model):
    #     """Hook function called before the start of each validation epoch."""
    #     self.scorer = scoring.build_scorer(self.args.scoring, self.tgt_dict)
        
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            # bleu

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

            # wer
            w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
            metrics.log_scalar("_w_errors", w_errors)
            wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
            metrics.log_scalar("_wv_errors", wv_errors)
            w_total = sum(log.get("w_total", 0) for log in logging_outputs)
            metrics.log_scalar("_w_total", w_total)

            if w_total > 0:
                metrics.log_derived(
                    "wer",
                    lambda meters: safe_round(
                        meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                    )
                    if meters["_w_total"].sum > 0
                    else float("nan"),
                )
                metrics.log_derived(
                    "raw_wer",
                    lambda meters: safe_round(
                        meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                    )
                    if meters["_w_total"].sum > 0
                    else float("nan"),
                )