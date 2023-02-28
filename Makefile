SRCDIR := src
INCDIR := include
EXTDIR := ext
BINDIR := target
DEPDIR := $(BINDIR)/deps

BIN := $(BINDIR)/cuda-image-transform

CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra -I $(INCDIR)/ -I $(EXTDIR)/FreeImage/build/include/ -L $(EXTDIR)/FreeImage/build/lib/ -fopenmp -g
CXXOFLAGS := -march=native -O3 
LDFLAGS := -lfreeimage -lm

NVCXX := nvcc
NVCXXFLAGS := -std=c++11 -I $(INCDIR)/ -I $(EXTDIR)/FreeImage/build/include/ -L $(EXTDIR)/FreeImage/build/lib/ -O3 -use_fast_math -extra-device-vectorization -Xcompiler -fopenmp -g
NVCXXARCH := -gencode arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY)

.PHONY: all clean

all: $(BIN)

$(BIN): $(DEPDIR)/config.o $(DEPDIR)/kernels.o $(DEPDIR)/utils.o $(DEPDIR)/driver.o $(SRCDIR)/main.cu
	@mkdir -p $(BINDIR)
	$(NVCXX) $(NVCXXFLAGS) $^ -o $@ $(LDFLAGS)

$(DEPDIR)/%.o: $(SRCDIR)/%.cu $(INCDIR)/%.cuh
	@mkdir -p $(DEPDIR)
	$(NVCXX) $(NVCXXARCH) $(NVCXXFLAGS) -c $< -o $@ $(LDFLAGS)

$(DEPDIR)/%.o: $(SRCDIR)/%.cpp $(INCDIR)/%.hpp
	@mkdir -p $(DEPDIR)
	$(CXX) $(CXXFLAGS) $(CXXOFLAGS) -c $< -o $@ $(LDFLAGS)

clean:
	@rm -rf $(BINDIR)
