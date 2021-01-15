#ifndef GLOBALS_H
#define GLOBALS_H
#include "raw.h"
#include <vector>

const int NUM_AXIS = 3;
const int WINDOW_SIZE = 10;
const int SAMPLING_RATE = 20;
const int NUM_TOKENS = 3 + NUM_AXIS;
const int AX1 = (int) 'X';
const int AX2 = (int) 'Y';
const int AX3 = (int) 'Z';
const int NUM_BINS = 10;
const int TOTAL_BINS = NUM_AXIS * NUM_BINS;
const int NUM_MFCC = 13;
const int TOTAL_MFCC = NUM_AXIS * NUM_MFCC;

extern int BIN_START;
extern int XMFCC_START;
extern int YMFCC_START;
extern int ZMFCC_START;
extern int coefIndex;
extern instancer chunk;
extern liner record;
extern std::vector<double> data[];

#endif