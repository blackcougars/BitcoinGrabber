// uint256 custom struct 
#include <iostream>

using namespace std;

struct uint256
{
    unsigned char value[64];
    uint256& operator++()
    {
        int i = 64;
        while (i > 0)
        {
            i -= 1;
            if (+*(this->value+i) < 0xF )
            {
                *(this->value+i) += 0x1;
                break;
            }            
        }   
        return *this; // return new value by reference
    }
    // uint256& operator<<()
    // {
    //     // actual increment takes place here
    //     int i = 32;
    //     while (i > 0)
    //     {
    //         i -= 1;
    //         if (+*(this->value+i) < 255 )
    //         {
    //             *(this->value+i) += 1;
    //             break;
    //         }
            
    //     }
    //     return *this; // return new value by reference
    // }
    friend ostream& operator << (ostream& out, const uint256 value)
    {
        //cout << +(value.value + 0) << "s" <<  endl;
        for (int i = 0; i < 64; i++)
            out << +(value.value[i]) << " ";
        out << std::endl;
        return out;
    }
};

// uint256 operator+ (const uint256 a, const uint256 b)
// {
//   //  uint256 result = {a.lh + b.lh + 2, a.rh + b.rh};
//     uint256 result;
//     result = {{'a','b'}, 12,2};
//     return result;
// }

// uint256 operator++(uint256 a)
// {
//     uint256 result;
//     int i = 32;
//     while (i > 0)
//     {
//         i -= 1;
//         if (+a.value[i] < 255 )
//         {
//             a.value[i] += 1;
//             break;
//         }
        
//     }
//     return result;
// }

int main()
{
    //cout << +a1 << endl;
    uint256 a;
    for (int i = 0; i < 64; i++)
        a.value[i] = 0xF;
    a.value[4] = 0xA;
    cout << a  << endl;
    ++a;
    //++a;
    cout << a << endl;
    //std::cout << c.lh << " " << c.rh << std::endl;
    return 0;
}