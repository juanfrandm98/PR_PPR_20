CPAR=./codigos_paralelo
CSEQ=./codigos_secuencial

all: bbseq bbpar

bbpar: ${CPAR}/bbpar.o ${CPAR}/libbb.o
	mpicxx -O3 ${CPAR}/bbpar.o ${CPAR}/libbb.o -o bbpar

bbseq: ${CSEQ}/bbseq.o ${CSEQ}/libbb.o
	mpicxx -O3 ${CSEQ}/bbseq.o ${CSEQ}/libbb.o -o bbseq

${CPAR}/bbpar.o: ${CPAR}/bbpar.cc
	mpicxx -O3 -c ${CPAR}/bbpar.cc -o ${CPAR}/bbpar.o

${CSEQ}/bbseq.o: ${CSEQ}/bbseq.cc
	mpicxx -O3 -c ${CSEQ}/bbseq.cc -o ${CSEQ}/bbseq.o

${CPAR}/libbb.o: ${CPAR}/libbb.cc ${CPAR}/libbb.h
	mpicxx -O3 -c  ${CPAR}/libbb.cc -o ${CPAR}/libbb.o

${CSEQ}/libbb.o: ${CSEQ}/libbb.cc ${CSEQ}/libbb.h
	mpicxx -O3 -c  ${CSEQ}/libbb.cc -o ${CSEQ}/libbb.o


clean:
	/bin/rm -f *.o bbseq
