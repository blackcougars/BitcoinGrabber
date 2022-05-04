#include <iostream>

#include "BigInt.h"

using namespace std;

int main ()
{
	BigInt a (1202020201023213321);
	BigInt b (3923042443214124233);
	
	BigInt c = a * b;
	cout << c << endl;
}
