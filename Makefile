# Use "make CUDA=1" for a CUDA build, use "make DEBUG=1" for a debug build

# The build artifacts are put in the "build-release" subfolder (or "build-debug" for a debug build).

# On Windows invoke with "make exe" or "make all"

DEBUG = 0
CUDA = 0
STATIC_RUNTIME = 0
STATIC_CUDA = 0

# Uncomment below as desired to set a particular compiler or force a debug build:
# CXX = g++-12
# DEBUG = 1
# or export those into environment, or pass on the command line e.g.
# make all DEBUG=1 CXX=g++-12

HOST_OS = $(shell uname -s)

CXX ?= g++

ifeq ($(CUDA), 1)
 BIN=build-cuda
 CUDASRCS1 = clwrap_cuda.cpp cudawrap.cpp
 CUDAFLAGS = -DCUDA_BACKEND -Isrc/cuda -I/usr/local/cuda/include
 CUDAOBJS = $(CUDASRCS1:%.cpp=$(BIN)/%.o)
 ifeq ($(STATIC_CUDA), 1)
  OPENCL_LIBS = -L/usr/local/cuda/lib64 -Wl,--start-group -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static -Wl,--end-group -lcuda -lpthread -ldl
 else
  OPENCL_LIBS = -L/usr/local/cuda/lib64 -Wl,-rpath,'$$ORIGIN' -lnvrtc -lcuda -lpthread
 endif
else
 BIN=build-release
 CUDAFLAGS =
 CUDAOBJS =
 ifeq ($(HOST_OS), Darwin)
  OPENCL_LIBS = -framework OpenCL
 else
  OPENCL_LIBS = -lOpenCL -lpthread
 endif
endif

COMMON_FLAGS = -Wall -Wextra $(CUDAFLAGS) -std=c++20

ifeq ($(STATIC_RUNTIME),1)
 LDFLAGS += -static-libstdc++ -static-libgcc

 ifeq ($(findstring MINGW, $(HOST_OS)), MINGW)
# For mingw-64 use this:
  LDFLAGS += -static
 endif
endif

ifeq ($(findstring MINGW, $(HOST_OS)), MINGW)
 CPPFLAGS += -DWINVER=0x0601 -D_WIN32_WINNT=0x0601
 LDFLAGS += -Wl,--subsystem,console:6.01
endif
# -fext-numeric-literals

ifeq ($(DEBUG), 1)

BIN=build-debug
CXXFLAGS = -g -Og $(COMMON_FLAGS)

else

CXXFLAGS = -O3 -flto -DNDEBUG $(COMMON_FLAGS)

endif

SRCS1 = fs.cpp Trig.cpp TuneEntry.cpp Primes.cpp tune.cpp CycleFile.cpp TrigBufCache.cpp Event.cpp Queue.cpp TimeInfo.cpp Profile.cpp bundle.cpp Saver.cpp KernelCompiler.cpp Kernel.cpp gpuid.cpp File.cpp Proof.cpp log.cpp Worktodo.cpp common.cpp main.cpp Gpu.cpp clwrap.cpp Task.cpp timeutil.cpp Args.cpp state.cpp Signal.cpp FFTConfig.cpp AllocTrac.cpp sha3.cpp md5.cpp version.cpp

SRCS2 = test.cpp

# SRCS=$(addprefix src/, $(SRCS1))

OBJS = $(CUDAOBJS) $(SRCS1:%.cpp=$(BIN)/%.o)
DEPDIR := $(BIN)/.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
POSTCOMPILE = @mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d && touch $@

all: prpll

prpll: $(BIN)/prpll

amd: $(BIN)/prpll-amd

#$(BIN)/test: $(BIN)/test.o
#	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBPATH)

$(BIN)/prpll: ${OBJS}
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ ${OBJS} $(LIBPATH) $(OPENCL_LIBS)

# Instead of linking with libOpenCL, link with libamdocl64
$(BIN)/prpll-amd: ${OBJS}
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ ${OBJS} $(LIBPATH) -lamdocl64 -L/opt/rocm/lib

clean:
	rm -rf build-debug build-release build-cuda

$(BIN)/%.o : src/%.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

$(BIN)/%.o : src/cuda/%.cpp $(DEPDIR)/%.d
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

# src/bundle.cpp is just a wrapping of the OpenCL sources (*.cl) as a C string (as well as the CUDA OpenCL translation code)

src/bundle.cpp: genbundle.sh src/cuda/*.cuh src/cl/*.cl
	bash genbundle.sh $^ > src/bundle.cpp

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

src/version.cpp : src/version.inc

src/version.inc: FORCE
	echo \"`basename \`git describe --tags --long --dirty --always --match v/prpll/*\``\" > $(BIN)/version.new
	diff -q -N $(BIN)/version.new $@ >/dev/null || mv $(BIN)/version.new $@
	echo Version: `cat $@`

FORCE:

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS1))))
# include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS2))))
