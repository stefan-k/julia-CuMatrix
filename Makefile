BIT=""
ifeq ($(shell uname -m), x86_64)
	BIT=64
endif


CXX=g++
CUDADIR=/usr/local/cuda
CUDALIB=$(CUDADIR)/lib$(BIT)
CFLAGS+=-I$(CUDADIR)/include
CXXFLAGS=$(CFLAGS)
LDFLAGS+=-shared -Wl,-soname,libcumem.so -o libcumem.so
LDFLAGS+=-L$(CUDALIB) -lcudart -Wl,-rpath,$(CUDALIB)
LDFLAGS+=-fPIC
SRC+=$(shell ls *.cpp)

all: libcumem.so

libcumem.so: $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC)

clean:
	rm -f *.o *.so
