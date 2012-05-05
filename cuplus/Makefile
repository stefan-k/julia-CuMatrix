BIT=""
ifeq ($(shell uname -m), x86_64)
	BIT=64
endif

TARGET=../libcuplus.so

CXX=g++
CUDADIR=/usr/local/cuda
CUDALIB=$(CUDADIR)/lib$(BIT)
CFLAGS+=-I$(CUDADIR)/include
CXXFLAGS=$(CFLAGS)
LDFLAGS+=-shared -Wl,-soname,libcuplus.so -o $(TARGET)
LDFLAGS+=-L$(CUDALIB) -lcuda -lcudart -lcurand -lcufft -Wl,-rpath,$(CUDALIB)
LDFLAGS+=-fPIC
SRC+=$(shell ls *.cpp)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(SRC)

clean:
	rm -f *.o $(TARGET)
