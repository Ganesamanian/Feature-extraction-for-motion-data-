#ifndef ARFF_H
#define ARFF_H
#include <sstream>
#include <fstream>
#include <string>
#include "funcmap.h"
#include "except.h"

class arff {
public:
    arff  (char **, void(*)(arff&), void(*)(arff&));
    ~arff ();
    void add (int, void(*)(const std::unique_ptr<attribute>&), std::string, std::string = "numeric");
    void setRelation (std::string _relation) { relation=_relation; }
    void compute ();
    std::ifstream rawStream;
private:
    void openFiles (char **) throw (OpenFileFail);
    funcmap * container;
    std::string relation;
    std::ofstream arfStream;
    std::stringstream instanceStream;
};

#endif