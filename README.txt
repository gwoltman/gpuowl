PRPLL "purrple cat[egorizer]"

CUDA execution tuning (integer optimization build)
--------------------------------------------------

An ordinary ./prpll -tune now measures and saves GRAPH_BATCH, GRAPH_TAIL,
L2PERSIST, L2PERSISTPCT, PDL and PDL_EARLY automatically. For M31/M61 transforms
it also searches NTT_WEIGHT_TABLE, INPLACE_TILE31/61 and NTT_TRANSPOSE31/61.
GRAPHS is searched together with batch size. No RTX 5090-specific values need
to be entered before tuning on another machine.

To retune just these settings with an existing installation:
  ./prpll -fft 1:512:7:512 -tune execution

The ordinary PRP, LL, checkpoint and proof workflows are unchanged. The new
settings are appended to config.txt, including explicit disabled values to
override previous settings. The execution-only command leaves tune.txt alone.
The usual -tune noconfig, regs, variants and fine modes retain their scope.

Timing uses real serial graph batches, with graph construction before the
timer. Candidates undergo Gerbicz checking. Two grouped search passes are
followed by a longer ABBA comparison with the starting settings; unverified
or slower results are not selected. This is a bounded empirical search,
not a guarantee of the globally fastest configuration. Run tuning on an idle
GPU and rerun it after changing GPU, driver or the main transform geometry.

Compatibility: early PDL notification together with repeated child graphs
failed full-residue checks on the RTX 5090. Tuning tests early notification only
with GRAPH_BATCH=1. Repeated batches retain ordinary late PDL notification.
The runtime also forces PDL_EARLY=0 for a manually requested incompatible pair
and logs the adjustment. The dependency wait includes a compiler memory barrier.

Build
-----

To build PRPLL use "make".

In the gpuowl project directory (where the file Makefile is located) run make.
This will produce a file "prpll" in the build-debug or build-release subdirectory.

Use "make" to do a release build in the "build-release" subdirectory.
Use "make CUDA=1" to produce a CUDA version in the "build-cuda" subdirectory.
Use "make DEBUG=1" to produce a debug build in the "build-debug" subdirectory.
Use "make exe" for a Windows build.
Use "make clean" to remove the "build-debug" and "build-release" directories.

Build example:

$ git clone git@github.com:preda/gpuowl.git
$ cd gpuowl
$ make
$ build-release/prpll -h


Run
----

1. run "clinfo" and see whether it finds any OpenCL devices. If clinfo does not find any devices, 
you need to fix your OpenCL installation first.

2. run "prpll -h", and verify that it displays a list of devices towards the end.

3. run "prpll -tune", to tune various PRPLL options for your GPU.

