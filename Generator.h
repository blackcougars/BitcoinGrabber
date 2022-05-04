#ifndef H_GENERATOR
#define H_GENERATOR


#define CUDA_MEMBER __host__        // Расположение функций класса (device или host)


#include <string>
#include <fstream>


using namespace std;


class Generator
{
    public:
        CUDA_MEMBER Generator();
        CUDA_MEMBER void start(string* progress, string* dbPath);
        CUDA_MEMBER string* stop();

    private:
        CUDA_MEMBER void startKernel();  // Метод запуска ядра на GPU
        CUDA_MEMBER void preparationData();
        //CUDA_MEMBER string progress;
        //CUDA_MEMBER string dbPath;
        CUDA_MEMBER string* arrayDataPtr;
        CUDA_MEMBER string* progressPtr;
};


#endif