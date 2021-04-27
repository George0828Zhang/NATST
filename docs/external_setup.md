# Setup for External Packages
## fast_align
```bash
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build
cd build
cmake ..
make
```

## mosesdecoder
See detailed instruction [here](http://www.statmt.org/moses/?n=Development.GetStarted) if you fail to compile with the following.
```bash
git clone https://github.com/moses-smt/mosesdecoder.git
cd mosesdecoder
./bjam -j4
```

## SimulEval
```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
git checkout eb748f7
pip install ./
```

## TER
```bash
mkdir tercom
cd tercom
wget http://www.cs.umd.edu/~snover/tercom/tercom-0.7.25.tgz
tar xzf tercom-0.7.25.tgz
```