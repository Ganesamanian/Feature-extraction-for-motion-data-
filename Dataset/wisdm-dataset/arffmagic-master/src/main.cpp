#include "try.h"
#include "except.h"
#include "read.h"
#include "write.h"
#include "arff.h"

void validateArgs (int argc) throw (ArgumentFail) {
    if (argc != 3) {
        throw ArgumentFail();
    }
}

int main (int argc, char ** argv) {
    tryOrDie([&]() {
        validateArgs(argc); 
        arff magic(argv, initribute, rawdog);
    });
}

