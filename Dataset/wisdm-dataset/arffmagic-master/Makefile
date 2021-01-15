SHELL=/bin/sh
SRC=./src
BIN=./bin
BUILD=./build
CC=clang++
CFLAGS=-std=c++11 -g -O2 -Wall -Wextra
LDFLAGS=-lAquila -lgsl -lgslcblas -lm
EXE=$(BIN)/arffmagic
OBJS=$(BUILD)/main.o $(BUILD)/globals.o $(BUILD)/raw.o $(BUILD)/read.o $(BUILD)/chunk.o $(BUILD)/write.o $(BUILD)/arff.o $(BUILD)/funcmap.o $(BUILD)/attribute.o $(BUILD)/libmfcc.o
DS:=dsymutil
DSYM=$(BIN)/arffmagic.dSYM
.SUFFIXES: 
.SUFFIXES: .c .cpp .o

dsym: $(DSYM)
	
all: $(EXE) $(DSYM)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS)

$(BUILD)/%.o: $(SRC)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(DSYM): $(EXE)
	$(DS) -o $@ $<

.PHONY: clean
clean: 
	rm -rf $(OBJS) $(EXE) $(DSYM)