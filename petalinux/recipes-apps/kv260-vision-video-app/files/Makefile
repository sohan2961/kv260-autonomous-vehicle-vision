APP = kv260-vision-video-app

APP_OBJS = kv260-vision-video-app.o

CXXFLAGS += -std=c++17 -Wall -Wextra
LDLIBS += -lvart-runner -lxir -lunilog

all: build

build: $(APP)

$(APP): $(APP_OBJS)
	$(CXX) -o $@ $(APP_OBJS) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(APP) *.o

# OpenCV 4
PKG_CONFIG ?= pkg-config
CXXFLAGS += $(shell $(PKG_CONFIG) --cflags opencv4)
LDLIBS += $(shell $(PKG_CONFIG) --libs opencv4)
