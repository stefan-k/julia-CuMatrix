BIT=""
ifeq ($(shell uname -m), x86_64)
	BIT=64
endif


CXX=g++
CUDADIR=/usr/local/cuda
CUDALIB=$(CUDADIR)/lib$(BIT)
CFLAGS+=-I$(CUDADIR)/include
CXXFLAGS=$(CFLAGS)
LDFLAGS+=-shared -Wl,-soname,libcuplus.so -o libcuplus.so
LDFLAGS+=-L$(CUDALIB) -lcudart -lcurand -Wl,-rpath,$(CUDALIB)
LDFLAGS+=-fPIC
SRC+=$(shell ls *.cpp)

all: libcuplus.so

libcuplus.so: $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC)

clean:
	rm -f *.o *.so
