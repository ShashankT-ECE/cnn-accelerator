# cnn_host — KV260 PS host application build
#
# Cross-compile for the Kria KV260 ARM Cortex-A53 (aarch64):
#   make CROSS_COMPILE=aarch64-linux-gnu-
#
# Native x86 build (compilation smoke test only; no UIO device present):
#   make native
#
# The cross toolchain is provided by gcc-aarch64-linux-gnu on Ubuntu 22.04:
#   sudo apt-get install gcc-aarch64-linux-gnu

CROSS_COMPILE ?= aarch64-linux-gnu-
CC     = $(CROSS_COMPILE)gcc
CFLAGS = -O2 -Wall -Wextra -std=c11

TARGET = cnn_host

all: $(TARGET)

$(TARGET): cnn_host.c
	$(CC) $(CFLAGS) -o $@ $^

native: cnn_host.c
	gcc $(CFLAGS) -o $(TARGET) $^

clean:
	rm -f $(TARGET)

.PHONY: all native clean
