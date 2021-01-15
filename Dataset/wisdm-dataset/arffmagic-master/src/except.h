#ifndef EXCEPT_H
#define EXCEPT_H
#include <stdexcept>

class EpicFail: public std::invalid_argument {
public:
    char buf[4096];
    EpicFail (const char * msg): std::invalid_argument(msg){};
    virtual const char * specify () {
        return specify("error", "god knows...");
    }
    virtual const char * specify (const char * errorLeft, const char * errorRight) {
        sprintf(buf, "%s: \"%s\"\n", errorLeft, errorRight);
        return buf;
    }
    virtual const char * specify (const char * errorLeft, const char * errorRight, const char * errorHint) {
        sprintf(buf, "%s: \"%s\"\n(%s)\n", errorLeft, errorRight, errorHint);
        return buf;
    }
};

class ArgumentFail: public EpicFail {
public:
    virtual const char * specify () {
        sprintf(buf, "%s: %-s\n%7s%-8s: %-s\n%7s%-8s: %-s\n", "USAGE", "arffmagic infile outfile", "", "infile", "raw data (.txt)", "", "outfile", "generated instances (.arff)");
        return buf;
    }
    ArgumentFail(): EpicFail(specify()) {}; 
};

class OpenFileFail: public EpicFail {
public:
    OpenFileFail(const char * file): EpicFail(specify("failed to open", file)) {};
};

class AttributeTypeFail: public EpicFail {
public:
    AttributeTypeFail(const char * type): EpicFail(specify("unknown type", type, "can only magic on numeric, nominal, and string attributes")) {};
};

#endif