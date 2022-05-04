#include "Generator.h"
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    // Проверка аргументов
    string progress = "";
    string dbPath = ""; // Пустой аргумент на будущее
    for (int i = 1; i < argc; i++)
    {   
        bool have = false;
        if ( string(argv[i]) == "--progress" && (i + 1) <= argc )
        {
            progress = string(argv[i+1]);
            have = true;
            i += 1;
            continue;
        }
        if ( string(argv[i]) == "--db" && (i + 1) <= argc )
        {
            dbPath = string(argv[i+1]);
            have = true;
            i += 1;
            continue;
        }

        if (!have)
        {
            cout << "attribute: " << string(argv[i]) << " not defined!" << endl;
            return -1;
        }
            
    }	
    //Generator* gb = new Generator();
    //gb->start(&progress);
    cout << progress << endl;
    return 0;
}
