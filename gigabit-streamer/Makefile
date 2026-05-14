# Makefile for Moku:Delta Gigabit Streamer tools

CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
LDFLAGS  := -lpthread -lm

.PHONY: all clean install-rt

all: send_it capture_it

# ── Transmitter ───────────────────────────────────────────────────────────────
send_it: send_it_buffer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built send_it — run with: sudo ./send_it --file capture.bin --fs 20833333 --dest <IP>"

# ── Capture (high-performance C++ — replaces captureDataBuffer.py) ────────────
capture_it: capture_it.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built capture_it — run with: sudo ./capture_it --outfile capture.bin --seconds 5 --verbose"

# Set SCHED_FIFO capability on both binaries so they can use real-time
# scheduling without sudo each time (Linux only).
install-rt: send_it capture_it
	sudo setcap cap_sys_nice+ep ./send_it
	sudo setcap cap_sys_nice+ep ./capture_it
	@echo "RT capability set on send_it and capture_it."

clean:
	rm -f send_it capture_it
