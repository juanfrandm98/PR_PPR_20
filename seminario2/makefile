####################################################
# MAKEFILE PARA LOS EJERCICIOS DEL SEMINARIO 2     #
# Programación Paralela                            #
# Juan Francisco Díaz Moreno                       #
# Abril de 2020                                    #
####################################################

SRC = src
BIN = bin
CXX = mpiCC

all: $(BIN)/p1 $(BIN)/p1_1 $(BIN)/p2 $(BIN)/p3 $(BIN)/p4

$(BIN)/p1 : $(SRC)/send_receive.cpp
	$(CXX) $(SRC)/send_receive.cpp -o $(BIN)/p1

$(BIN)/p1_1 : $(SRC)/send_receive_1.cpp
	$(CXX) $(SRC)/send_receive_1.cpp -o $(BIN)/p1_1

$(BIN)/p2 : $(SRC)/calculo_pi.cpp
	$(CXX) $(SRC)/calculo_pi.cpp -o $(BIN)/p2

$(BIN)/p3 : $(SRC)/producto_escalar.cpp
	$(CXX) $(SRC)/producto_escalar.cpp -o $(BIN)/p3

$(BIN)/p4 : $(SRC)/comunicadores.cpp
	$(CXX) $(SRC)/comunicadores.cpp -o $(BIN)/p4

clean :
	rm -f $(BIN)/*
