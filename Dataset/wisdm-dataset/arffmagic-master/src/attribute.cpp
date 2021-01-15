#include "attribute.h"

bool attribute::spaceFound (string str) {
    unsigned long i = 0; bool space = false;
    while (!space && i<str.length()) {
        space = isspace(str[i]);
        i++;
    } return space;
}

string attribute::representStr (string str) {
    string newstr = str;
    if (spaceFound(str)) {
        newstr = "\"";
        newstr += (str += "\"");
    } return newstr;
}

string nominal::representType (string str) {
    string newstr = "{ ";
    newstr += (str += " }");
    return newstr;
}

void nominal::set (string str) {
    val = representStr(str);
    if (types.empty()) {
        types.push_back(val);
        type += val;
    } else if ((std::find(types.begin(), types.end(), val)) == types.end()) {
        types.push_back(val);
        type += ", ";
        type += val;
    }
}

void numeric::print (std::ofstream& strm) const { strm << val; }
void numeric::print (std:: ostream& strm) const { strm << val; }
void nominal::print (std::ofstream& strm) const { strm << val; }
void nominal::print (std:: ostream& strm) const { strm << val; }
void stringy::print (std::ofstream& strm) const { strm << val; }
void stringy::print (std:: ostream& strm) const { strm << val; }

attribute::~attribute () {};
numeric::~numeric () {};
nominal::~nominal () {};
stringy::~stringy () {};
