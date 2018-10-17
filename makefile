
HMLP_DEV_INC := -I${HMLP_DIR}/build/include/ -I${HMLP_DIR}/frame -I${HMLP_DIR}/gofmm -I${HMLP_DIR}/frame/base -I${HMLP_DIR}/frame/containers -I${HMLP_DIR}/frame/mpi -I${HMLP_DIR}/kernel/x86_64/skx -I${HMLP_DIR}/kernel/reference -I${MKLROOT}/include
HMLP_MASTER_INC := -I${HMLP_DIR}/build/include/ -I${HMLP_DIR}/frame/ -I${HMLP_DIR}/frame/mpi -I${HMLP_DIR}/kernel/mic/knl -I${HMLP_DIR}/kernel/reference -I${MKLROOT}/include
CC := icpc # This is the main compiler
# CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
TARGET := bin/runner


SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -Wall -O3 -std=c++11 -fPIC -pedantic -DUSE_INTEL -DUSE_BLAS -DUSE_VML -mkl=parallel -qopenmp -xMIC-AVX512 -DHMLP_MIC_AVY512
LIB := -L/lib64/ -lpthread ${MKLROOT}/../compiler/lib/intel64/libiomp5.so -L${HMLP_DIR}/build/lib/ -lhmlp -Wl,-rpath,${HMLP_DIR}/build/lib
INC := -I./include ${HMLP_MASTER_INC} 

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"; $(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

# Tests
tester:
	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

# Spikes
ticket:
	$(CC) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket

.PHONY: clean
