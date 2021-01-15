#include "read.h"
#include "globals.h"
#include "chunk.h"
#include "write.h"

void getRecord (arff& barf) {
    getline(barf.rawStream, record.curr);
    record.tokenize();
}

void rawdog (arff& barf) {
    while (barf.rawStream.good()) {
        bool instance = false; bool discard = false;
        if (validRecord()) {
            if (belongsInChunk()) {
                if (!chunk.full()) {
                    addRecord();
                } else {
                    instance = true;
                }
            } else if (!chunk.min()) {
                discard = true;
            } else {
                instance = true;
            }
        } if (instance) {
            writeInstance(barf);
        } else if (!discard) {
            record.update();
            getRecord(barf);
        }
    }
}
