# CUDA code generation flags

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SMXX    := -gencode arch=compute_70,code=compute_70
GENCODE_FLAGS   ?= $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SMXX)

floyd: floyd.cu Graph.cc Graph.h
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS)   floyd.cu Graph.cc -o floyd

floyd2d: floyd2d.cu Graph.cc Graph.h
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS)   floyd2d.cu Graph.cc -o floyd2d

floyd2dreduccion: floyd2dreduccion.cu Graph.cc Graph.h
	nvcc -I./includes  -O3 -m64   $(GENCODE_FLAGS)   floyd2dreduccion.cu Graph.cc -o floyd2dreduccion

clean:
	rm -f floyd *.o.
