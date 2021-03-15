
CFLAGS=-I /usr/local/neuware/include/ -g -O0  -std=c++11
CXXFLAGS=$(CFLAGS)

LDFALGS=-L /usr/local/neuware/lib64 `pkg-config --libs opencv` 

CXX=g++

SRCS=$(wildcard *.cpp)
OBJS=$(SRCS:%.cpp=%.o)

TARGET=yolov5_offline_simple_demo

all:$(TARGET)

$(TARGET):$(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFALGS) -lcnrt -lcndrv -lpthread 
	
clean:
	rm  -f $(TARGET) $(OBJS)

