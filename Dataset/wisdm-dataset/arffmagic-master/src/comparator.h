#ifndef COMPARATOR_H
#define COMPARATOR_H

class comparator {
public:
    bool operator () (const std::unique_ptr<attribute>& left, const std::unique_ptr<attribute>& right) const {
        return left->position < right->position;
    }
};

#endif