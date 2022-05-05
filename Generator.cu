#include "Generator.h"
#include "Kernel.h"


#include <iostream>

 
CUDA_MEMBER Generator::Generator()
{
    
}

CUDA_MEMBER int Generator::preparationData(string* progress, string* dbPath)
{
    // Подготовка данных к запуску
    
    // Подготовка массива ключей/адресов
    if (!dbPath)
        dbPath = new string ("db.txt");

    ifstream* file = new ifstream(*dbPath);
    if(!file->is_open())
        // Ошибка открытия файла 
        return -1;
    string data;
    int countData = 0;
    while(*file >> data)
        countData++;
    file->clear();
    file->seekg(0);
    string* arrayData = new string[countData];
    int place = 0;
    while(*file >> data)
        arrayData[place] = data;

    // Выделение памяти на GPU
    string* arrayDataPtr;
    string* progressPtr;
    long int* countDataPtr;
    cudaMalloc((void**)&arrayDataPtr, sizeof(string) * countData);
    cudaMalloc((void**)&progressPtr, sizeof(string));
    cudaMalloc((void**)&countDataPtr, sizeof(long int));


    // Загрузка данных в память GPU
    cudaMemcpy(arrayDataPtr, arrayData, sizeof(string) * countData, cudaMemcpyHostToDevice);
    cudaMemcpy(progressPtr, progress, sizeof(string), cudaMemcpyHostToDevice);
    cudaMemcpy(countDataPtr, &countData, sizeof(long int), cudaMemcpyHostToDevice);

    this->countDataPtr = countDataPtr;
    this->arrayDataPtr = arrayDataPtr;
    this->progressPtr = progressPtr;

    return 0;
}

CUDA_MEMBER int Generator::start(string* progress, string* dbPath)
{
    // Подготовка к запуску ядра
    int statusCode = this->preparationData(progress, dbPath);
    if (statusCode != 0)
        return statusCode;
    this->startKernel();

    return 0;
}

CUDA_MEMBER string* Generator::stop()
{
    // Остановка ядра, сохранение прогресса, высвобождение ресурсов
    string* progress = new string("none");// Не реализовано

    
    // Высвобождение ресурсов GPU  
    cudaFree(this->arrayDataPtr);
    cudaFree(this->progressPtr);

    return progress;
}

CUDA_MEMBER void Generator::startKernel()
{
    // Запуск ядра     
    dim3 gridSize = dim3(1, 1, 1);      // Размер используемого грида
    dim3 blockSize = dim3(1024, 1, 1);      
    kernel <<<gridSize, blockSize>>> (this->arrayDataPtr, this->countDataPtr, this->progressPtr);
}