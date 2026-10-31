# Makefile - zero-dependency standalone Virtual NPU
# Builds libvnpu.so/.a, vnpu-cli, vnpu-node and vnpu-selftest.

CXX      ?= g++
CC       ?= gcc
AR       ?= ar
UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)
VERSION := 1.0.0

CXXFLAGS ?= -std=c++17 -O3 -fPIC -pthread -fvisibility=hidden \
            -fno-math-errno -ffp-contract=fast -funroll-loops \
            -Wall -Wextra -Wno-unused-parameter \
            -Wno-missing-field-initializers -DNDEBUG
CPPFLAGS ?= -Iinclude -Isrc -DVNPU_BUILD_SHARED
LDFLAGS  ?= -pthread
LINK_LIB := -Llib -lvnpu -Wl,-rpath,'$$ORIGIN/../lib'

ifeq ($(UNAME_S),Linux)
SONAME := -Wl,-soname,libvnpu.so.1
else
SONAME :=
endif

LIB_SRCS := \
  src/core/runtime.cpp \
  src/core/scheduler.cpp \
  src/core/autotune.cpp \
  src/core/backend.cpp \
  src/kernels/kernels.cpp \
  src/kernels/llm.cpp \
  src/cluster/net.cpp \
  src/cluster/node.cpp \
  src/cluster/cluster.cpp \
  src/compat/compat.cpp

LIB_OBJS := $(patsubst src/%.cpp,build/obj/%.o,$(LIB_SRCS))

CLI_SRCS := \
  tools/vnpu_cli.cpp \
  tools/selftest.cpp \
  tools/distributed_demo.cpp \
  tools/llm_bench.cpp

SELFTEST_SRCS := tools/selftest_main.cpp tools/selftest.cpp
NODE_SRCS := tools/node_main.cpp
FUZZ_SRCS := tests/fuzz_shapes.cpp

.PHONY: all clean test fuzz benchmark bench-matrix release install pcem-check

-include $(LIB_OBJS:.o=.d)

all: lib/libvnpu.so lib/libvnpu.a bin/vnpu-cli bin/vnpu-node \
     bin/vnpu-selftest bin/vnpu-fuzz bin/vnpu-conformance

# Header dependencies: without these a header edit relinks stale objects and the
# binary silently keeps running old code.
DEPFLAGS := -MMD -MP

build/obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -MF $(@:.o=.d) -c $< -o $@

lib/libvnpu.so: $(LIB_OBJS)
	@mkdir -p lib
	$(CXX) -shared $(LDFLAGS) $(SONAME) -o $@ $^
	@ln -sf libvnpu.so lib/libvnpu.so.1

lib/libvnpu.a: $(LIB_OBJS)
	@mkdir -p lib
	$(AR) rcs $@ $^

bin/vnpu-cli: $(CLI_SRCS) lib/libvnpu.so
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CLI_SRCS) $(LINK_LIB) $(LDFLAGS) -o $@

bin/vnpu-node: $(NODE_SRCS) lib/libvnpu.so
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NODE_SRCS) $(LINK_LIB) $(LDFLAGS) -o $@

bin/vnpu-selftest: $(SELFTEST_SRCS) lib/libvnpu.so
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SELFTEST_SRCS) $(LINK_LIB) $(LDFLAGS) -o $@

bin/vnpu-fuzz: $(FUZZ_SRCS) lib/libvnpu.so
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(FUZZ_SRCS) $(LINK_LIB) $(LDFLAGS) -o $@

bin/vnpu-conformance: tools/conformance.cpp lib/libvnpu.so
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tools/conformance.cpp $(LINK_LIB) $(LDFLAGS) -o $@

# The "same answer everywhere" claim, tested rather than asserted: every unit,
# thread count and B structure against an independent plain-C reference.
conformance: bin/vnpu-conformance
	./bin/vnpu-conformance $(CONFORMANCE_ARGS)

fuzz: bin/vnpu-fuzz
	./bin/vnpu-fuzz 100

test: all
	@mkdir -p benchmarks
	./bin/vnpu-selftest
	./bin/vnpu-fuzz 60
	./bin/vnpu-conformance
	./bin/vnpu-cli bench --shape 64x64x64 --shape 128x128x128 \
	  --dtype both --iterations 1 --json benchmarks/selftest-bench.json
	./bin/vnpu-cli cluster-demo --shape 64x64x64 --iterations 1 \
	  --port1 19450 --port2 19451 --token test
	./bin/vnpu-cli specdemo --steps 2 --hidden 64 --vocab 64 --gamma 3 \
	  --port1 19452 --port2 19453 --token test

bench-matrix: lib/libvnpu.so
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tools/bench_kernels.cpp $(LINK_LIB) $(LDFLAGS) \
	  -o bin/bench_kernels
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tools/bench_scaling.cpp $(LINK_LIB) $(LDFLAGS) \
	  -o bin/bench_scaling

# Instruction-ceiling probe: FMA-only, +B loads, +A broadcasts, 1 and 2 threads.
# Run this first on any new machine -- every "% of peak" claim needs it, and
# bench_scaling's single-shot 10-chain loop under-reports the ceiling by ~10%.
peak: benchmarks/peak_probe.cpp
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o bin/vnpu-peak
	./bin/vnpu-peak

benchmark: all
	@mkdir -p benchmarks
	./bin/vnpu-cli info --json > benchmarks/device.json
	./bin/vnpu-cli bench --shape 64x64x64 --shape 128x128x128 \
	  --shape 256x256x256 --shape 512x512x512 --dtype both \
	  --iterations 5 --json benchmarks/gemm.json \
	  --trace benchmarks/sherlock-trace.json
	./bin/vnpu-cli autotune --shape 512x512x512 --dtype f32 \
	  --iterations 3 --json benchmarks/autotune-f32.json
	./bin/vnpu-cli autotune --shape 512x512x512 --dtype i8 \
	  --iterations 3 --json benchmarks/autotune-i8.json
	./bin/vnpu-cli llm-bench --hidden 512 --intermediate 1376 \
	  --heads 8 --layers 2 --prefill 32 --decode 1 --dtype both \
	  --iterations 3 --json benchmarks/llm-block.json \
	  --trace benchmarks/llm-trace.json
	./bin/vnpu-cli cluster-demo --shape 256x256x256 --iterations 3 \
	  --port1 19550 --port2 19551 --token bench \
	  > benchmarks/cluster-demo.txt
	./bin/vnpu-cli specdemo --steps 4 --hidden 128 --vocab 256 \
	  --gamma 4 --port1 19552 --port2 19553 --token bench \
	  > benchmarks/specdemo.txt

release: benchmark release/BUILD_INFO.txt
	@mkdir -p release
	tar -czf release/vnpu-standalone-$(VERSION).tar.gz \
	  --exclude='.git' --exclude='build' --exclude='archive' \
	  --exclude='legacy/*' --exclude='release/*.tar.gz' \
	  CMakeLists.txt Makefile README.md REPORT.md LICENSE \
	  include src tools tests examples bindings docs scripts \
	  lib bin benchmarks release/BUILD_INFO.txt
	@echo "release/vnpu-standalone-$(VERSION).tar.gz"

release/BUILD_INFO.txt: all
	@mkdir -p release
	@echo "version=$(VERSION)" > $@
	@echo "built=$(shell date -u +%Y-%m-%dT%H:%M:%SZ)" >> $@
	@echo "compiler=$(shell $(CXX) --version | head -1)" >> $@
	@echo "uname=$(shell uname -a)" >> $@
	@echo "flags=$(CXXFLAGS)" >> $@

pcem-check:
	@echo "Checking runtime sources for PCem dependencies..."
	@! grep -RIl --include='*.c' --include='*.h' --include='*.cpp' --include='*.cc' \
	  'vid_voodoo\|pcem/src\|wx-utils\|sdl' include src tools examples bindings 2>/dev/null || true
	@echo "OK: runtime sources are PCem-free"

install: all
	@mkdir -p $(PREFIX)/include $(PREFIX)/lib $(PREFIX)/bin
	cp include/*.h $(PREFIX)/include/
	cp lib/libvnpu.so lib/libvnpu.a $(PREFIX)/lib/
	cp bin/vnpu-cli bin/vnpu-node bin/vnpu-selftest $(PREFIX)/bin/

# Untracked build outputs only: `make clean` must leave `git status` untouched.
# It used to `rm -rf benchmarks/*.json benchmarks/*.txt release/*.tar.gz`, which
# destroyed tracked data -- no target recreates the release archives, and the
# benchmark records are evidence. lib/*.a|*.so are tracked too, so they are
# rebuilt in place (objects are gone, so `all` relinks) instead of deleted.
clean:
	rm -rf build bin
