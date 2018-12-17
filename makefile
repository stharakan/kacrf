
HMLP_DEV_INC := -I${HMLP_DIR}/build/include/ -I${HMLP_DIR}/frame/ -I${HMLP_DIR}/gofmm -I${HMLP_DIR}/frame/mpi -I${HMLP_DIR}/frame/containers -I${HMLP_DIR}/frame/base -I${HMLP_DIR}/kernel/x86_64/skx -I${HMLP_DIR}/kernel/reference -I${MKLROOT}/include -I${KACRF_DIR}/src
#HMLP_MASTER_INC := -I${HMLP_DIR}/build/include/ -I${HMLP_DIR}/frame/ -I${HMLP_DIR}/frame/mpi -I${HMLP_DIR}/kernel/mic/knl -I${HMLP_DIR}/kernel/reference -I${MKLROOT}/include
CC := icpc # This is the main compiler
#CC := clang --analyze # and comment out the linker last line for sanity
SRCDIR := src
BUILDDIR := build
TARGET := bin/runner

#CC := mpicxx
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -g -Wall -O3 -std=c++11 -fPIC -pedantic -DUSE_INTEL -DUSE_BLAS -DUSE_VML -DHMLP_USE_MPI -mkl=parallel -qopenmp -xMIC-AVX512 -DHMLP_MIC_AVY512 
CFLAGS := -g -O3 -qopenmp -Wall -pedantic -fPIC -D_POSIX_C_SOURCE=200112L -DUSE_INTEL -DUSE_BLAS -DUSE_VML -mkl=parallel -xCORE-AVX2 -axCORE-AVX512,MIC-AVX512 -std=c++11
LIB := -L/lib64/ -lpthread ${MKLROOT}/../compiler/lib/intel64/libiomp5.so -L${HMLP_DIR}/build/lib/ -lhmlp -Wl,-rpath,${HMLP_DIR}/build/lib
LIB := -Xlinker -rpath -Xlinker /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib/release_mt -Xlinker -rpath -Xlinker /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib -Xlinker -rpath -Xlinker /opt/intel/mpi-rt/2017.0.0/intel64/lib/release_mt -Xlinker -rpath -Xlinker /opt/intel/mpi-rt/2017.0.0/intel64/lib -Wl,-rpath,/home1/03158/tharakan/lib/hmlp-develop/build/lib:/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib:/opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib/release_mt ${HMLP_DIR}/build/lib/libhmlp.so /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib/libmpicxx.so /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib/libmpifort.so /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib/release_mt/libmpi.so /opt/intel/compilers_and_libraries_2017.4.196/linux/mpi/intel64/lib/libmpigi.a -ldl -lrt -lpthread ${MKLROOT}/../compiler/lib/intel64/libiomp5.so

INC := -I./include ${HMLP_DEV_INC} 

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

tkspa:
	$(CC) $(CFLAGS) test/testkspa.cpp $(INC) $(LIB) -o bin/kspa-test

bcrf:
	$(CC) $(CFLAGS) test/brain_crf.cpp $(INC) $(LIB) -o bin/bcrf

cvcrf:
	$(CC) $(CFLAGS) test/cv_brain_crf.cpp $(INC) $(LIB) -o bin/cvcrf

# Spikes
ticket:
	$(CC) $(CFLAGS) spikes/ticket.cpp $(INC) $(LIB) -o bin/ticket

.PHONY: clean
