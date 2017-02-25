INCLUDE_DIRS =  -I/usr/local/opencv/include
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lrt
CPPLIBS= -L/usr/local/opencv/lib -lopencv_core -lopencv_flann -lopencv_video

HFILES= 
CFILES= 

SRCS= ${HFILES} ${CFILES}

#all:	capture stereo_match capture_stereo
all:	capture 

clean:
	-rm -f *.o *.d
	-rm -f capture
#	-rm -f stereo_match

distclean:
	-rm -f *.o *.d


capture: capture.o
	$(CC) $(LDFLAGS) $(CFLAGS) $(INCLUDE_DIRS) -o $@ $@.o `pkg-config --libs opencv` $(CPPLIBS)

#stereo_match: stereo_match.o
#	$(CC) $(LDFLAGS) $(CFLAGS) $(INCLUDE_DIRS) -o $@ $@.o `pkg-config --libs opencv` $(CPPLIBS)

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CC) $(CFLAGS) -c $<
