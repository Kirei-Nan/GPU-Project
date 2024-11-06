# Makefile for compiling the SimpleModel project

# Compiler
CXX = g++

# Compilation flags
CXXFLAGS = -std=c++11 -Wall

# Source files
SRCS = model.cpp main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Header files
HEADERS = model.h

# Executable name
TARGET = simple_model

# Default rule
all: $(TARGET)

# Linking step
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compilation step
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove compiled files
clean:
	rm -f $(OBJS) $(TARGET)
