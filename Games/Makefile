# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -I/opt/homebrew/include -I/opt/homebrew/include/SDL2 -IML-Models/NN -Ilibs -Ilibs/Data-Processing -Ilibs/Math


# Linker flags
LDFLAGS = -L/opt/homebrew/lib -lSDL2 -lSDL2_ttf

# Target executable
TARGET = pong

# Source files
SOURCES = pong.cpp \
          ML-Models/NN/NeuralNetwork.cpp \
          ML-Models/NN/Neuron.cpp \
          ML-Models/NN/NetworkLayers.cpp \
          libs/Math/ActivationFunctions.cpp \
          libs/Math/LinearAlgebra.cpp \
          libs/Math/GenFunctions.cpp \
          libs/Math/OptimizationMethods.cpp \
          libs/Math/NeuroHelperFuncs.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default rule to build the target
all: $(TARGET)

# Rule to link all object files into the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

# Rule to compile each .cpp file into a .o file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -f $(OBJECTS) $(TARGET)
