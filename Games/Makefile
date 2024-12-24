# Compiler
CXX = g++

CXXFLAGS = -std=c++11 -Wall -I/opt/homebrew/include -I/opt/homebrew/include/SDL2

LDFLAGS = -L/opt/homebrew/lib -lSDL2 -L/opt/homebrew/lib -lSDL2_ttf


# Target executable
TARGET = out

# Automatically detect all .cpp files in the project (including ML-Models)
SOURCES = pong.cpp

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
	rm -f $(OBJECTS) $(TARGET) out