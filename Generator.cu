#include "Generator.h"
#include <secp256k1.h>
#include <stdio.h>

__global__ void generator(unsigned long long int* startNumber, unsigned long int* countAddresses, std::string* arrayAddresses, unsigned long long int* countCheckedPtr)
{
    //atomicAdd(countCheckedPtr, 1);
    blockIdx.x ;
    if (threadIdx.x == 1)
        *countCheckedPtr += 1;
    // Generation key and check
    // unsigned int warp = 1'000'000;
    // *startNumber = *startNumber + ((threadIdx.x * warp) - warp); // Start number private key (256 bits)
    // unsigned long long int endNumber = *startNumber + warp;
    // unsigned long countThreads = 1;      // Count activing threads


    // while(*startNumber <= endNumber)
    // {
    //     // Private key to public key

    
    //     // Public key to payment addresses


    //     // Check address have in db
    //     int i = 0;
    //     while(i < *countAddresses)
    //     {
    //         if (true)
    //         {
                
    //         }
    //         i += 1;
    //     }
    //     if (*startNumber == endNumber)
    //         // Когда дошел до конца, увеличение
    //         endNumber = endNumber + countThreads * warp;
    //     *startNumber += 1;
    // }
    // endNumber = endNumber + countThreads * warp;

}

