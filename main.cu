/*
    * BlackCatGenerator
    * Программа для перебора закрытых ключей сети Bitcoin
    * Оптимизирована под видеокарту (Cuda)
    * Зависимости - нет
*/
//#include "Generator.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>


using namespace std;

__global__ void generator( unsigned long long int* countCheckedPtr)
{
    //atomicAdd(countCheckedPtr, 1);
    atomicAdd(countCheckedPtr, 1);

    //__syncthreads();
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


 __host__ int main ()
{
    unsigned long long int countChecked = 2;
    cout << "Starting..." << endl;
    ofstream* fileOut = new ofstream("valid.txt", ios::binary);
    ifstream* fileAddress = new ifstream("db.txt");
    
    // Read addresses
    string address;
    double balance;
    unsigned long int countAddresses = 0;
    // while(*fileAddress >> address >> balance)
    //     countAddresses++;
    string* arrayAddresses = new string[countAddresses];
    fileAddress->clear();
    fileAddress->seekg(0);
    unsigned long int place = 0;
    // while(*fileAddress >> address >> balance && place < 1000)
    // {   
    //     // Reading addresses from db
    //     *(arrayAddresses + place) = address;
    //     place++;
    // }
    //cout << cudaGetDeviceProperties << endl;
    unsigned long long int startCount = 0;
    int threadNum = 1024;
    dim3 blockSize = dim3(threadNum, 1, 1);
    dim3 gridSize = dim3(96, 1, 1);
    
    // Copying data to GPU memory
    unsigned long long int* startCountAddressPtr = nullptr;
    unsigned long int* countAddressesPtr = nullptr;
    unsigned long long int* countCheckedPtr = nullptr;
    string* arrayAddressesPtr = nullptr;
    cudaMalloc((void**)&startCountAddressPtr, sizeof(unsigned long long int));      
    cudaMalloc((void**)&countCheckedPtr, sizeof(unsigned long long int));      
    cudaMalloc((void**)&countAddressesPtr, sizeof(unsigned long int));      
    cudaMalloc((void**)&arrayAddressesPtr, sizeof(string) * countAddresses);      

    cudaMemcpy(startCountAddressPtr, &startCount, sizeof(unsigned long long int), cudaMemcpyHostToDevice);      
    cudaMemcpy(countCheckedPtr, &countChecked, sizeof(unsigned long long int), cudaMemcpyHostToDevice);      
    cudaMemcpy(countAddressesPtr, &countAddresses, sizeof(unsigned long int), cudaMemcpyHostToDevice);      
    cudaMemcpy(arrayAddressesPtr, arrayAddresses, sizeof(string) * countAddresses, cudaMemcpyHostToDevice);      
    // Starting core at GPU
    cout << countAddressesPtr << endl;
    generator<<<gridSize, blockSize>>>(countCheckedPtr);

    while(countChecked < (pow(2, sizeof(countChecked) * 8)))
    {
        cout << (pow(2, sizeof(countChecked) * 8)) << endl;
        
        // Check find priv key on GPU
        this_thread::sleep_for(chrono::milliseconds(500));    
        //unsigned long long int* countCheckedLocal = nullptr;
        cudaMemcpy(&countChecked, countCheckedPtr, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);      
        // Enter count checked priv key
        cout << countChecked << endl;
    }    
    // Free GPU memory
    cudaFree(startCountAddressPtr);
    cudaFree(countAddressesPtr);
    cudaFree(arrayAddressesPtr);
    return 0;
}

