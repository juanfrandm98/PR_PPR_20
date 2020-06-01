all: bbpar bbseq

bbpar: bbpar.o libbb.o
	mpicxx -O3 bbpar.o libbb.o -o bbpar

bbseq: bbseq.o libbb.o
	mpicxx -O3 bbseq.o libbb.o -o bbseq

bbpar.o: bbpar.cc
	mpicxx -O3 -c bbpar.cc

bbseq.o: bbseq.cc
	mpicxx -O3 -c bbseq.cc

libbb.o: libbb.cc libbb.h
	mpicxx -O3 -c  libbb.cc


clean:
	/bin/rm -f *.o bbseq
