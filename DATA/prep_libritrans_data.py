#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Tuple
import csv
import pandas as pd
import torchaudio
from data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,


    load_df_from_tsv,
    filter_manifest_df,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import walk_files
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]

class LibriTrans(Dataset):

    SPLITS = [
        "other",
        "train",
        "dev",
        "test",
    ]
    METACOLS = "Book_id|Chapter_id|reader_id|original_librispeech_subset|"
    "audio_filename|duration".split("|")

    _ext_audio = ".wav"

    def __init__(self, root: str, split: str) -> None:
        assert split in self.SPLITS
        _root = Path(root) / split
        wav_root, txt_root = _root / "audiofiles", _root #_root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        # segments = load_df_from_tsv(_root / "alignments.meta")    
        segments = pd.read_csv(
            (_root / "alignments.meta").as_posix(),
            sep="\t",
            skiprows = 1,
            header = None,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
            na_filter=False,
        )
        
        # Load source and target utterances
        with open(txt_root / f"{split}.en") as f:
            src_texts =  [r.strip() for r in f]
        with open(txt_root / f"{split}.fr") as f:
            tgt_texts =  [r.strip() for r in f]
        
        assert len(segments) == len(src_texts)
        assert len(segments) == len(tgt_texts)
        # import pdb
        # pdb.set_trace()

        # Gather info
        self.data = []
        for i, d in enumerate(segments.values):
            book_id, chapter_id, reader_id, libriset, wav_filename, duration = d
            wav_path = wav_root / f"{wav_filename}{self._ext_audio}"
            try:                
                ret = torchaudio.info(wav_path.as_posix())
            except RuntimeError:
                continue
            # sample_rate = ret.sample_rate
            # n_frames = int(float(duration) * sample_rate)
            _id = f"{wav_path.stem}"
            self.data.append(
                (
                    wav_path.as_posix(),                    
                    src_texts[i],
                    tgt_texts[i],
                    reader_id,
                    _id,
                )
            )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        wav_path, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, sr = torchaudio.load(wav_path)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    out_root = Path(args.data_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    failed = {}
    for split in LibriTrans.SPLITS:
        print(f"Fetching split {split}...")
        dataset = LibriTrans(out_root.as_posix(), split)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            try:
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy"
                )
            except AssertionError:
                failed[utt_id] = 1
    # Pack features into ZIP
    zip_path = out_root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in LibriTrans.SPLITS:
        is_train_split = split in ("train", "other")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = LibriTrans(out_root.as_posix(), split)
        for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            if utt_id in failed:
                continue
            manifest["id"].append(utt_id)
            manifest["audio"].append(zip_manifest[utt_id])
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            if args.lowercase:
                src_utt = src_utt.lower()
                tgt_utt = tgt_utt.lower()
            manifest["src_text"].append(src_utt)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["src_text"])
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, out_root / f"{split}_{args.task}.tsv")
        
    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            out_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        out_root, 
        spm_filename_prefix + ".model", 
        yaml_filename=f"config_{args.task}.yaml",
        specaugment_policy="ld"
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
