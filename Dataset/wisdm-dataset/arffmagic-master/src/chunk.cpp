#include "chunk.h"
#include "globals.h"
#include "read.h"

bool validRecord () {
    return (record.curr != record.prev) && (record.parsed == record.expected);
}

bool belongsInChunk () {
    return (chunk.empty()) || (!strcmp(record.toke[0], record.oldToke[0])) || (!strcmp(record.toke[1], record.oldToke[1]));
}

void addRecord () {
    for (int i=0; i<NUM_AXIS; i++) {
        data[i].push_back(atof(record.toke[i+3]));
    }
    chunk.resize(chunk.size()+1);   
}
void dumpChunk () {
    for (int i=0; i<NUM_AXIS; i++) {
        data[i].clear();
    }
    chunk.resize(0);
}