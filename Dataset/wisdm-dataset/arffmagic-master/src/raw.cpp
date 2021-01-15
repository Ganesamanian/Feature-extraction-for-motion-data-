#include "raw.h"

instancer::instancer (int x): valid(0), invalid(0), MAX(x), MIN(0.9*x), count(0) {};

liner::liner (int x): expected(x), parsed(0), toke(NULL), oldToke(NULL), prev(""), curr("") {
    size_t size = sizeof(char **) * (expected + 1);
    toke = (char **) malloc(size);
    oldToke = (char **) malloc(size);
    memset(toke, '\0', size);
    memset(oldToke, '\0', size);
}

void liner::update () {
    prev = curr;
    for (int i=0; i<expected; i++) {
        if (toke[i]) {
            free(oldToke[i]);
            oldToke[i] = toke[i];
        }
    }
}

void liner::tokenize () {
    parsed = 0;
    const char * c_str = curr.c_str(); 
    char * dup = strdup(c_str); 
    char * token = strtok(dup, ",;:");
    while (token && parsed<expected) {
        toke[parsed] = strdup(token);
        parsed++;
        token = strtok(NULL, ",;:");
    } free(dup);
}

liner::~liner () {
    free(toke);
    free(oldToke);
}
