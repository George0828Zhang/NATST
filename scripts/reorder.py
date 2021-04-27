#!/usr/bin/env python

import argparse
import numpy as np

def reorder(inputs):
    """
    srcseq, tgtseq: ["tokens", ...]
    alnseq: ["0-0", "6-0", ...]
    """
    srcseq, tgtseq, alnseq = inputs
    tlen = len(tgtseq)
    null = -1
    new_order = np.full(tlen, null)
    for s_t in alnseq:
        s, t = tuple(map(int, s_t.split('-')))  # (0,0)
        new_order[t] = s
    
    for i in range(tlen):
        if new_order[i] == null:
            new_order[i] = new_order[i-1] if i > 0 else 0

    # gauranteed to be stable
    reordered = [x for _, x in sorted(zip(new_order, tgtseq), key=lambda pair: pair[0])]
    return reordered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="source file")
    parser.add_argument("-t", "--target", type=str, help="target file")
    parser.add_argument("-a", "--align", type=str, help="alignment file")
    parser.add_argument("-o", "--output", type=str, help="output file")
    args = parser.parse_args()
    print(args)

    def readlines(name):
        t = []
        with open(name, "r") as f:
            for line in f:
                t.append(line.strip().split())
        return t

    srcs = readlines(args.source)
    tgts = readlines(args.target)
    alns = readlines(args.align)

    assert len(srcs) == len(tgts) and len(alns) == len(tgts)

    # single cpu
    results = map(reorder, zip(srcs, tgts, alns))

    with open(args.output, "w") as f:
        for line in results:
            f.write(" ".join(line)+"\n")
