.SILENT:


TARGET=streambox
LIB_SRC=avtuner.cpp\
	detector.cpp

OBJECTS=$(LIB_SRC:.cpp=.o) $(TARGET).o
LIBS=$(LIB_SRC:.cpp=.a)

CXX=g++-4.9
LIBXX=ar
LXX=ld
CXXFLAGS=-Wno-deprecated-declarations --std=c++11 -L/usr/local/lib
DEPLIBS=tesseract\
	avfilter\
	swresample\
	opencv_core\
	opencv_imgproc\
	opencv_highgui\
	opencv_gpu\
	pthread\
	z\
	x264\
	va\
	dl\
	vdpau\
	X11\
	va-drm\
	va-x11\
	lzma

FFMPEGLIB=avformat\
	avcodec\
	swscale\
	avutil

all: $(OBJECTS) $(LIBS) $(TARGET)

.cpp.o:
	echo $(CXX) $(CXXFLAGS) -c $^
	$(CXX) $(CXXFLAGS) -c $^

.o.a:
	$(LIBXX) rcs $@ $^

$(TARGET):
	echo $(CXX) $(CXXFLAGS) $@.cpp -o $@ $(LIBS) $(addprefix -l, $(DEPLIBS))
	$(CXX) $(CXXFLAGS) $@.o -o $@ $(LIBS) $(addprefix -l, $(DEPLIBS)) $(addprefix -l, $(FFMPEGLIB))
	strip $@
#	bzip2 -kf --best $@
#	$(LXX) $@.o -o $@ $(LIBS) $(addprefix -l, $(DEPLIBS)) -lstdc++ -lm -lsocket++

rm:
	rm -f *.o
	rm -f *.a
	rm -f $(TARGET)
	rm -f *.exe
	rm -f 4.flv
