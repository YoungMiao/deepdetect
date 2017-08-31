#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include<iostream>

using namespace std;
 
typedef struct preResult
{
	vector<double> prob;
	//double prob;
    vector<double> xmax;
    vector<double> ymax;
    vector<double> ymin;
    vector<double> xmin;

}preResult;

typedef struct classesName
{
    string className;
	vector<preResult> vec;
}classesName;

#endif