#ifndef H_KERNEL
#define H_KERNEL

#include <string>

using namespace std;

extern __global__ void kernel(string* arrayData, long int* countData, string* progress);


#endif