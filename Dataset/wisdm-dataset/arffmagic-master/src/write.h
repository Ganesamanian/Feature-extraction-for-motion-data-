#ifndef WRITE_H
#define WRITE_H
#include "attribute.h"
#include "arff.h"
#include <vector>
using std::vector;
// process writing to the arff file

void initribute (arff&);
//axis * 
int getAxis (const std::unique_ptr<attribute>&, int);
void activity (const std::unique_ptr<attribute>&);
void bin (const std::unique_ptr<attribute>&);
int getBindex (const std::unique_ptr<attribute>&);
void average  (const std::unique_ptr<attribute>&);
void peak (const std::unique_ptr<attribute>&);
vector<int> getPeaks (const std::unique_ptr<attribute>&);
vector<int> getDists (vector<int>);
void absoluteDev (const std::unique_ptr<attribute>&);
void standardDev (const std::unique_ptr<attribute>&);
void variance (const std::unique_ptr<attribute>&);
void mfcc (const std::unique_ptr<attribute>&);
void cosineSimilarity (const std::unique_ptr<attribute>&);
void correlation (const std::unique_ptr<attribute>&);
void resultant (const std::unique_ptr<attribute>&);
double square (double);
void userID (const std::unique_ptr<attribute>&);
void writeInstance (arff&);


#endif