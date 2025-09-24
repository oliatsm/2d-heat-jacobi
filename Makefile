# compiler
CC = nvc

# compiler flags
CCFLAGS = -fast -Mprof=ccff

#l

.PHONY : clean
clean :
        rm edit $(objects)