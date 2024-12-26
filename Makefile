# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude -Iinclude/libs -Iinclude/libs/Math -Iinclude/libs/Data-Processing -Iinclude/ML -Iinclude/ML/NN -I/opt/homebrew/include -I/opt/homebrew/include/SDL2


LDFLAGS = -L/opt/homebrew/lib -lSDL2 -lSDL2_ttf

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(wildcard $(SRC_DIR)/main/Main.cpp) $(wildcard $(SRC_DIR)/ML/NN/*.cpp) $(wildcard $(SRC_DIR)/libs/Math/*.cpp) $(wildcard $(SRC_DIR)/libs/Data-Processing/*.cpp)

# Object files
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SOURCES))

# Target executable
TARGET = $(BIN_DIR)/trading_system

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
