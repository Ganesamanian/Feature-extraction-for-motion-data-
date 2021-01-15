#ifndef TRY_H
#define TRY_H
#include "except.h"
#include <iostream>
template <typename x>
void tryOrDie (x func) {
    try {
        func();
    } catch (const EpicFail& wtf) {
        std::cerr << wtf.what();
        exit(1);
    }
}
#endif