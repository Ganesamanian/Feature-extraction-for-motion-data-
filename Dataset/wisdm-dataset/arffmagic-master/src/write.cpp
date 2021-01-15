#include "write.h"
#include "read.h"
#include "chunk.h"
#include "globals.h"
#include "aquila/aquila.h"
#include "libmfcc.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include <cmath>

int binOffset = 0; 
int mfccOffset = 0;

double avg [NUM_AXIS];
double sdev[NUM_AXIS];
int bins[NUM_AXIS][NUM_BINS];
vector<double> coefs[NUM_AXIS];

void initribute (arff& barf) {
    barf.setRelation("person_activities_labeled");

    /*  ADDING ATTRIBUTES TO ARFFMAGIC!
    for each attribute desired, make a call to the following arff member function
        add(<int>, <void(*)(const std::unique_ptr<attribute>&)>, <string>, <string>)
    with corresponding parameters 
        (1) position in instance (AKA position in arff header/attribute declaration)
        (2) function (that you need to define) to set the attribute's value
        (3) name (used for attribute declaration)
        (4--optional) type (defaults to "numeric", alternatively specify "string" or "nominal") 
    Only attributes added in this function will appear in the generated arff! */

    int n = 1;
    barf.add(n++, activity, "ACTIVITY", "nominal");

    binOffset = n;
    for (int ax = AX1; ax < (NUM_AXIS + AX1); ax++) {
        int start = NUM_BINS * (ax-AX1);
        string str = "";
        str += (char)ax;
        for (n = start; n < (start + NUM_BINS); n++) {
            barf.add(n+binOffset, bin, str+(std::to_string(n % NUM_BINS)));
        } n += (TOTAL_BINS - n) + (ax - AX1) + binOffset;
        barf.add(n, average, str + "AVG");
        barf.add(n+=NUM_AXIS, peak, str + "PEAK");
        barf.add(n+=NUM_AXIS, absoluteDev, str + "ABSOLDEV");
        barf.add(n+=NUM_AXIS, standardDev, str + "STANDDEV");
        barf.add(n+=NUM_AXIS, variance, str + "VAR");
    }

    mfccOffset = n+1;
    for (int ax = AX1; ax < (NUM_AXIS + AX1); ax++) {
        int start = NUM_MFCC * (ax-AX1);
        string str = "";
        str += (char)ax;
        str += "MFCC";
        for (n = start; n < (start + NUM_MFCC); n++) {
            barf.add(n+mfccOffset, mfcc, str+(std::to_string(n % NUM_MFCC)));
        } n += (TOTAL_MFCC - n) + (ax - AX1) + mfccOffset;
    }

    barf.add(++n, cosineSimilarity, "XYCOS");
    barf.add(++n, cosineSimilarity, "XZCOS");
    barf.add(++n, cosineSimilarity, "YZCOS");
    barf.add(++n, correlation, "XYCOR");
    barf.add(++n, correlation, "XZCOR");
    barf.add(++n, correlation, "YZCOR");
    barf.add(++n, resultant, "RESULTANT");
    barf.add(++n, userID, "class", "nominal");
}

int getAxis (const std::unique_ptr<attribute>& attr, int nameChar) {
    int axis;
    switch (attr->name[nameChar]) {
        case 'X': axis = 0; break;
        case 'Y': axis = 1; break;
        case 'Z': axis = 2; break;
    } 
    return axis;
}

void activity (const std::unique_ptr<attribute>& attr) {
    attr->set(record.toke[1]);
}

void bin (const std::unique_ptr<attribute>& attr) {
    int ax = getAxis(attr, 0);
    attr->set((bins[ax][getBindex(attr)])/((double)chunk.size()));
}

int getBindex (const std::unique_ptr<attribute>& attr) {
    int bindex = ((attr->position)-binOffset)%NUM_BINS;
    return bindex;
}

void average (const std::unique_ptr<attribute>& attr) {
    int ax = getAxis(attr, 0);
    attr->set(avg[ax]);
}

void peak (const std::unique_ptr<attribute>& attr) {
    int sum = 0;
    vector<int> dists = getDists(getPeaks(attr));
    for (auto dist: dists) {
        sum += dist;
    } 
    attr->set(((double)sum/dists.size())*NUM_BINS);
}

vector<int> getPeaks (const std::unique_ptr<attribute>& attr) {
    int ax = getAxis(attr, 0);
    vector<int> peaks;
    if (data[ax][0] > data[ax][1]) {
        peaks.push_back(0);
    } for (int i=1; i<(chunk.size()-2); i++) {
        if ((data[ax][i] > data[ax][i-1]) && data[ax][i] > data[ax][i+1]) {
            peaks.push_back(i);
        }
    } 
    if (data[ax][chunk.size()-1] > data[ax][chunk.size()-2]) {
        peaks.push_back(chunk.size()-1);
    } 
    return peaks;
}

vector<int> getDists (vector<int> peaks) {
    vector<int> dists;
    for (unsigned long i=0; i<(peaks.size()-2); i++) {
        dists.push_back(peaks[i+1] - peaks[i]);
    } 
    return dists;
}

void absoluteDev (const std::unique_ptr<attribute>& attr) {
    double dev = 0.0;
    int ax = getAxis(attr, 0);
    for (auto var: data[ax]) {
        dev += std::abs(var-(avg[ax]));
    } 
    attr->set(dev/(double)chunk.size());
}

void standardDev (const std::unique_ptr<attribute>& attr) {
    int ax = getAxis(attr, 0);
    attr->set(sdev[ax]);
}

void variance (const std::unique_ptr<attribute>& attr) {
    int ax = getAxis(attr, 0);
    attr->set(sqrt(sdev[ax]));
}

int getCoefIndex (const std::unique_ptr<attribute>& attr) {
    int coefIndex = ((attr->position)-mfccOffset)%NUM_MFCC;
    return coefIndex;
}

void mfcc (const std::unique_ptr<attribute>& attr) {
    int ax = getAxis(attr, 0);
    attr->set(coefs[ax][getCoefIndex(attr)]);   
}

void cosineSimilarity (const std::unique_ptr<attribute>& attr) {
    vector<double> * left = &data[getAxis(attr, 0)];
    vector<double> * right = &data[getAxis(attr, 1)];
    double top = 0.0;
    double bottomLeft = 0.0;
    double bottomRight = 0.0;
    for (size_t i=0; i<(*left).size(); i++) {
        top += (((*left)[i])*((*right)[i]));
        bottomLeft  += pow((* left)[i], 2);
        bottomRight += pow((*right)[i], 2);
    } 
    attr->set(top/((sqrt(bottomLeft))*(sqrt(bottomRight))));
}

void correlation (const std::unique_ptr<attribute>& attr) {
    vector<double> * left = &data[getAxis(attr, 0)];
    vector<double> * right = &data[getAxis(attr, 1)];
    gsl_vector_const_view gslLeft = gsl_vector_const_view_array((*left).data(), (*left).size());
    gsl_vector_const_view gslRight = gsl_vector_const_view_array((*right).data(), (*right).size());
    attr->set(gsl_stats_correlation(gslLeft.vector.data, 1, gslRight.vector.data, 1, (*left).size()));
}

// template <typename x>
// void iterateChunk (x func) {
//     for (int j=0; j<chunk.size(); j++) {
//         func(j);
//     }
// }

void resultant (const std::unique_ptr<attribute>& attr) {
    double sum = 0.0;
    for (int i=0; i<chunk.size(); i++) {
        double squared = 0.0;
        for (int j=0; j<NUM_AXIS; j++) {
            squared += pow(data[j][i], 2);
        }
        sum += sqrt(squared);
    }
    attr->set(sum/((double)chunk.size()));
}


void userID (const std::unique_ptr<attribute>& attr) {
    attr->set(record.toke[0]);
}

int binDrop (double var) {
    int bindex = 0;
    var = (double) var/2.5;
    if (var >= -1 && var <= 7) {
        bindex = (floor(var))+2; 
    } else if (var > 7) {
        bindex = 9;
    } 
    return bindex;
}

vector<double> getFFT (vector<double> data) {
    size_t before = data.size();
    while ((data.size() & (data.size()-1)) != 0) {
        data.push_back(0.0);
    }
    Aquila::SignalSource source(data, SAMPLING_RATE);
    Aquila::AquilaFft fft(source.getSamplesCount());
    auto spectrum = fft.fft(source.toArray());
    std::vector<double> realFFT;
    for (auto complex: spectrum) {
        realFFT.push_back(std::abs(complex));
    }
    chunk.resize(before);
    return realFFT;
}

void writeInstance (arff& barf) {
    for (int i=0; i<NUM_AXIS; i++) {
        avg[i] = sdev[i] = 0.0;
        coefs[i].clear();
        memset(bins[i], 0, sizeof(int) * NUM_BINS);
        for (auto val: data[i]) {
            bins[i][binDrop(val)]++;
            avg[i] += val;
        }
        avg[i] /= chunk.size();
        for (auto val: data[i]) {
            sdev[i] += pow((val-avg[i]), 2);
        }
        sdev[i] = (sqrt(sdev[i]))/((double)chunk.size());
        vector<double> FFT = getFFT(data[i]);
        for (int j=0; j<NUM_MFCC; j++) {
            coefs[i].push_back(GetCoefficient(FFT.data(), SAMPLING_RATE, 48, FFT.size(), j));
        }
    }
    barf.compute();
    dumpChunk();
    chunk.valid++;
}
