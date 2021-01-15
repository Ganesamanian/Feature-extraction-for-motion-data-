#ifndef FUNCMAP_H
#define FUNCMAP_H
#include <fstream>
#include <vector>
#include <map>
#include "attribute.h"
#include "comparator.h"
#include "except.h"

class funcmap {
public:
    template <typename stream>
    friend stream& operator << (stream& strm, const funcmap * container) {
        for (auto& x: container->pairs) {
            strm << x.first;
            if (x != *(container->pairs.rbegin())) {
                strm << ",";
            }
        } return strm;
    }
    void push (int, string, string, void(*)(const std::unique_ptr<attribute>&)) throw (AttributeTypeFail);
    void hash ();
    void writeHeader (std::ofstream&);
private:
    std::map<const std::unique_ptr<attribute>, void(*)(const std::unique_ptr<attribute>&), comparator> pairs;
};

#endif