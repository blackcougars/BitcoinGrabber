#include "Generator.h"
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    // Проверка аргументов
    string progress = "";
    string* dbPath = nullptr; // Пустой аргумент на будущее
    for (int i = 1; i < argc; i++)
    {   
        bool have = false;
        cout << "lol2" << endl;
        if ( string(argv[i]) == "--progress" && (i + 1) <= argc )
        {
            progress = string(argv[i+1]);
            have = true;
            i += 1;
            continue;
        }
        if ( string(argv[i]) == "--db" && (i + 1) <= argc )
        {
            
            dbPath = new string(argv[i+1]);
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

    cout << "lol" << endl;
    Generator* gb = new Generator();
    int statusCode = gb->start(&progress, dbPath);

    if (statusCode == -1)
    {
        cout << "Db file error!" << endl;
        return statusCode;
    }

        //cout << progress << endl;
    return 0;
}
