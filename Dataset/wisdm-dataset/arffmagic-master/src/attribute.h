#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H
#include <algorithm>
#include <fstream>
#include <vector>
#include <memory>
using std::string;

class attribute {
public:
    attribute (int _position, string _name, string _type): position(_position), name(_name), type(_type), title("@attribute") {};
    virtual ~attribute ();
    string declare () { 
        return (this->title + " \"" + this->name + "\" " + this->representType(this->type)); 
    }
    const int position;
    string name;
    string type;
    const string title;
    bool spaceFound (string);
    string representStr (string);
    virtual string representType (string str) { return str; }
    virtual void set (double) {};
    virtual void set (string) {};
    virtual void setType (std::vector<const char *>) {};
    virtual void print (std::ofstream&) const=0;
    virtual void print (std:: ostream&) const=0;
    template <typename stream>
    friend stream& operator << (stream& strm, const std::unique_ptr<attribute>& attr){
        attr->print(strm);
        return strm;
    }
};

class numeric: public attribute {
public:
    double val;
    numeric (int _position, string _name, string _type): attribute(_position, _name, _type), val(0.0) {}
    virtual ~numeric ();
    void set (double x) { val = x; }
    void print (std::ofstream&) const;
    void print (std:: ostream&) const;
};

class nominal: public attribute {
public:
    std::vector<string> types;
    string val;
    nominal (int _position, string _name, string _type): attribute(_position, _name, _type), val("") {};
    virtual ~nominal ();
    virtual string representType (string);
    void set (string);
    void print (std::ofstream&) const;
    void print (std:: ostream&) const;
};

class stringy: public attribute {
public:
    string val;
    stringy (int _position, string _name, string _type): attribute(_position, _name, _type), val("") {};
    virtual ~stringy ();
    void set (string x) { val = representStr(x); }
    void print (std::ofstream&) const;
    void print (std:: ostream&) const;
};

#endif 