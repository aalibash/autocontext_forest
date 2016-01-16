INCLUDES = -I/is/ps/shared/users/asrikantha/local/OpenCV-2.4.3/include -I/is/ps/shared/users/asrikantha/local/gsl-1.14 -I/is/ps/shared/users/asrikantha/local/libconfig-1.4.9/lib 
LIBS = -lopencv_core -lopencv_highgui -lconfig++ -lgsl -lgslcblas -lm
LIBDIRS = -L/is/ps/shared/users/asrikantha/local/OpenCV-2.4.3/lib -Wl,-rpath,/is/ps/shared/users/asrikantha/local/OpenCV-2.4.3/lib -L/is/ps/shared/users/asrikantha/local/libconfig-1.4.9/lib/.libs -Wl,-rpath,/is/ps/shared/users/asrikantha/local/libconfig-1.4.9/lib/.libs -L/is/ps/shared/users/asrikantha/local/gsl-1.14 -Wl,-rpath,/is/ps/shared/users/asrikantha/local/is/ps/shared/users/asrikantha/local/gsl-1.14


OPT = -O2 -Wno-deprecated

CC=g++

.PHONY: all clean

OBJS = Features.o HoG.o HFForest.o HFTree.o Node.o BinaryTest.o Hypotheses.o Param.o TrainingData.o AnnotationData.o FPFNmodel.o

clean:
	rm -f *.o *~ *.gch HFTrainDetect

all:    HFTrainDetect
	echo all: make complete


%.o:%.cpp
	$(CC) -c $(INCLUDES) $+ $(OPT)

HFTrainDetect: HFTrainDetect.o $(OBJS)
	$(CC) $(LIBDIRS) $(LIBS) -o $@ $+ $(OPT)
