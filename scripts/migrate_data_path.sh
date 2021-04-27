#!/usr/bin/env bash
ROOT=/home/george/stream-st/DATA/covost/es
from=/media/george/Data/covost
to=/home/george/stream-st/DATA/covost

for f in `ls ${ROOT}/*.tsv ${ROOT}/*.yaml`; do
	echo ${f}
	sed -i "s~$from~$to~g" $f
done
