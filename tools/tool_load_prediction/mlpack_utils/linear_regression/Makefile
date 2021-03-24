# Declare the targets, src
TARGET	:= lr_regression
SRC		:= lr_samoa_logs.cpp
LIBS_NAME := armadillo boost_serialization mlpack

CXX := g++
CXXFLAGS += -std=c++11 -Wall -Wextra -O3 -DNDEBUG -fopenmp
# Use these CXXFLAGS instead if you want to compile with debugging symbols and
# without optimizations.
# CXXFLAGS += -std=c++11 -Wall -Wextra -g -O0
LDFLAGS  += -fopenmp
LDFLAGS += -L/home/ctminh/projects/loc-libs/mlpack-3.4.2/install/lib # /path/to/mlpack/library/ if installed locally.
# Add header directories for any includes that aren't on the
# default compiler search path.
INCLFLAGS := -I/home/ctminh/projects/loc-libs/mlpack-3.4.2/install/include # -I/usr/include
CXXFLAGS += $(INCLFLAGS)

OBJS := $(SRC:.cpp=.o)
LIBS := $(addprefix -l,$(LIBS_NAME))
CLEAN_LIST := $(TARGET) $(OBJS)

# default rule
default: all

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LIBS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@echo CLEAN $(CLEAN_LIST)
	@rm -f $(CLEAN_LIST)