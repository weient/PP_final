CC = gcc
CXX = g++
NVCC = nvcc
HIPCC = hipcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -dlto
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a
LDFLAGS = -lpng -lm

SOURCES = main.cu image.cu filters.cu kernels.cu
TARGET = filter

# Default target
all: $(TARGET)

# Build the executable directly from sources
$(TARGET): $(SOURCES)
	$(NVCC) $(NVFLAGS) $(SOURCES) $(LDFLAGS) -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     : Build the filter program (default)"
	@echo "  clean   : Remove build artifacts"
	@echo "  help    : Display this help message"

.PHONY: all clean help