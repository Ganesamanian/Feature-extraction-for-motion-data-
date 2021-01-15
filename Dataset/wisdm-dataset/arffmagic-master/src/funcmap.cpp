#include "funcmap.h"
#include "try.h"

void funcmap::hash () {
    for (auto& x: pairs) { 
        x.second(x.first); 
    } 
}

void funcmap::writeHeader (std::ofstream& ofs) {
    for (auto& x: pairs) {
        ofs << x.first->declare() << std::endl;
    } 
    ofs << std::endl;
}

void funcmap::push (int position, string name, string type, void(*calc)(const std::unique_ptr<attribute>&)) throw (AttributeTypeFail) {
    std::unique_ptr<attribute> curr = nullptr;
    tryOrDie([this, &curr, &position, &name, &type, &calc]() { 
        if (type == "numeric") {
            curr = std::unique_ptr<attribute>(new numeric(position, name, type));
        } else if (type == "string") {
            curr = std::unique_ptr<attribute>(new stringy(position, name, type));
        } else if (type == "nominal") {
            curr = std::unique_ptr<attribute>(new nominal(position, name, ""));
        } else {
            throw AttributeTypeFail(type.c_str());
        } 
        pairs.insert(std::make_pair(std::move(curr), calc));
    });
}