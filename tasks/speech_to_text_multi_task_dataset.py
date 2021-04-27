# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import csv
# import io
import logging
import os.path as op
# import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    # Dictionary,
    # FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
# from fairseq.data.audio.audio_utils import get_fbank, get_waveform
# from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import (
    get_features_or_waveform,
    _collate_frames,
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator
)

logger = logging.getLogger(__name__)

class SpeechToTextMultiTaskDataset(SpeechToTextDataset):
    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        source = get_features_or_waveform(
            self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
        )
        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        asr_target = None
        if self.src_texts is not None:
            tokenized = self.tokenize_text(self.src_texts[index])
            asr_target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()

        return index, source, target, asr_target # transcript

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None

        def _collate_text(samples, key):
            assert key in (2,3), "Text index can only be 2 or 3 for tuple (id, frames, target, asr_target)"
            target = fairseq_data_utils.collate_tokens(
                [t[key] for t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t[key].size(0) for t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t[key] for t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t[key].size(0) for t in samples)
            return target, target_lengths, prev_output_tokens, ntokens

        if self.tgt_texts is not None:
            target, target_lengths, prev_output_tokens, ntokens = _collate_text(samples, 2)
        
        asr_target, asr_target_lengths, asr_prev_output_tokens = None, None, None
        asr_ntokens = None
        if self.src_texts is not None:
            asr_target, asr_target_lengths, asr_prev_output_tokens, asr_ntokens = _collate_text(samples, 3)

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
        return out

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len += len(tokenized.split(" "))
        if self.src_texts is not None:
            tokenized = self.tokenize_text(self.src_texts[index])
            t_len += len(tokenized.split(" "))
        return self.n_frames[index], t_len


class SpeechToTextMultiTaskDatasetCreator(SpeechToTextDatasetCreator):
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SpeechToTextMultiTaskDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        return SpeechToTextMultiTaskDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )
    
    # @classmethod
    # def from_tsv(
    #     cls,
    #     root: str,
    #     data_cfg: S2TDataConfig,
    #     splits: str,
    #     tgt_dict,
    #     pre_tokenizer,
    #     bpe_tokenizer,
    #     is_train_split: bool,
    #     epoch: int,
    #     seed: int,
    # ) -> SpeechToTextMultiTaskDataset:


    #     samples = []
    #     _splits = splits.split(",")
    #     for split in _splits:
    #         tsv_path = op.join(root, f"{split}.tsv")
    #         if not op.isfile(tsv_path):
    #             raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    #         with open(tsv_path) as f:
    #             reader = csv.DictReader(
    #                 f,
    #                 delimiter="\t",
    #                 quotechar=None,
    #                 doublequote=False,
    #                 lineterminator="\n",
    #                 quoting=csv.QUOTE_NONE,
    #             )
    #             samples.append([dict(e) for e in reader])
    #             assert len(samples) > 0

    #     datasets = [
    #         cls._from_list(
    #             name,
    #             is_train_split,
    #             [s],
    #             data_cfg,
    #             tgt_dict,
    #             pre_tokenizer,
    #             bpe_tokenizer,
    #         )
    #         for name, s in zip(_splits, samples)
    #     ]

    #     if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
    #         # temperature-based sampling
    #         size_ratios = cls._get_size_ratios(
    #             _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
    #         )
    #         datasets = [
    #             ResamplingDataset(
    #                 d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
    #             )
    #             for d, r in zip(datasets, size_ratios)
    #         ]
    #     return ConcatDataset(datasets)
