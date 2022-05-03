#include <string>
#include <fstream>
//#include <cuda/atomic>
#include <cuda.h>

__global__ extern void generator(unsigned long long int* startNumber, unsigned long int* countAddresses, std::string* arrayAddresses, unsigned long long int* countCheckedPtr);