# Compiler
CXX = g++

CXXFLAGS = -std=c++11 -Wall -Ilibs -Ilibs/Data-Processing -Ilibs/Math -IML-Models -I/opt/homebrew/include/SDL2

LDFLAGS = -L/opt/homebrew/lib -lSDL2


# Target executable
TARGET = out

# Automatically detect all .cpp files in the project (including ML-Models)
SOURCES = $(wildcard Main.cpp libs/*/*.cpp ML-Models/NeuralNetwork.cpp ML-Models/NetworkLayers.cpp ML-Models/Neuron.cpp )

# Generate object files from sources
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
	rm -f $(OBJECTS) $(TARGET) results.csv