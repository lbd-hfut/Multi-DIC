CXX ?= g++
AR ?= ar
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic

BUILD_DIR := build
INCLUDES := -Iinclude
SOURCES := src/ncorr_api.cpp
OBJECTS := $(SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
TARGET := $(BUILD_DIR)/libnative_ncorr.a
CLI_TARGET := $(BUILD_DIR)/ncorr_cli

.PHONY: all clean

all: $(TARGET) $(CLI_TARGET)

$(TARGET): $(OBJECTS)
	$(AR) rcs $@ $^

$(CLI_TARGET): tools/ncorr_cli.cpp $(TARGET)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(TARGET) -o $@

$(BUILD_DIR)/%.o: src/%.cpp include/ncorr_api.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
