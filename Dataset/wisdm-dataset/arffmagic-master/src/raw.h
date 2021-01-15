#ifndef RAW_H
#define RAW_H
#include <cstdlib>
#include <string>

class instancer {
public:
    int valid;
    int invalid;
    instancer (int);
    void resize (unsigned int size) { count = size; };
    int size () { return count; };
    bool empty () const { return count == 0; }
    bool min () const { return count >= MIN; };
    bool full () const { return count == MAX; };
private:
    const int MAX;
    const int MIN;
    int count;
};

class liner {
public:
    const int expected;
    int parsed;
    char ** toke;
    char ** oldToke;
    std::string prev;
    std::string curr;
    void tokenize ();
    void update ();
    liner  (int);
    ~liner ();
};

#endif
