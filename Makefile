# Put your CUDA PATH here :)
CUDA_PATH := /usr/local/cuda

NVCC := $(CUDA_PATH)/bin/nvcc

NVCCFLAGS := -O2 -std=c++11

TARGET := gpu_sim

OBJS := physics.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

physics.o: physics.cu physics.h
	$(NVCC) $(NVCCFLAGS) -c physics.cu -o physics.o

main.o: main.cu physics.h utils.h
	$(NVCC) $(NVCCFLAGS) -c main.cu -o main.o

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
