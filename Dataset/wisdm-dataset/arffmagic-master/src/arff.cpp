#include "arff.h"
#include "try.h"
using std::endl;

arff::arff (char ** argv, void(* init)(arff&), void(* translate)(arff&)): container(new funcmap), relation("") {
    tryOrDie([this, &argv]() { openFiles(argv); });
    init(*this);
    arfStream << "@relation " << relation << endl << endl;
    translate(*this);
    container->writeHeader(arfStream);
    arfStream << "@data" << endl << instanceStream.str();
    std::cout << "magic @ " << argv[2] << endl;
}

arff::~arff () {
    delete container;
    rawStream.close();
    arfStream.close();
}

void arff::openFiles (char ** argv) throw (OpenFileFail) {
    rawStream.open(argv[1]);
    if (!rawStream.is_open()) {
        throw OpenFileFail(argv[1]);
    } else {
        arfStream.open(argv[2]);
        if (!arfStream.is_open()) {
            throw OpenFileFail(argv[2]);
        }
    }
}

void arff::add (int position, void(*calc)(const std::unique_ptr<attribute>&), std::string name, std::string type) {
    container->push(position, name, type, calc);
}

void arff::compute () {
    container->hash();
    instanceStream << container << endl;
}
