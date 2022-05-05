#include "BigInt.h"
#include <iostream>

using namespace std;

__host__ BigInt inverseMod (BigInt* a, BigInt* p)
{
    if ( *a < 0 or *a >= *p) 
        *a = *a % *p;

    BigInt c = *a;
    BigInt d = *p;
    BigInt uc = 1;
    BigInt vc = 0;
    BigInt ud = 0;
    BigInt vd = 1;
    while (c > 0)
    {
        BigInt q = d / c;
        c = d % c;
        d = c;

        uc = ud - q * uc;
        vc = vd - q * vc;
        ud = uc;
        vd = vc;
    }
        
    if( ud > 0) 
        return ud;
    return ud + *p;
}


__host__ int main()
{   
   

    // secp256k1
    BigInt gX = BigInt("55066263022277343669578718895168534326250603453777594175500187360389116729240");
    BigInt gY = BigInt("32670510020758816978083085130507043184471273380659243275938904335757337482424");
    BigInt gOrder = BigInt("115792089237316195423570985008687907852837564279074904382605163141518161494337");
    BigInt P = BigInt("115792089237316195423570985008687907853269984665640564039457584007908834671663");
    BigInt pubKey;
    BigInt qX = gX;
    BigInt qY = gY;
    bool have = false;
    BigInt e = BigInt("83020463237742095447187601505567755285685578327720972455401612915394172527322");  // private key
    BigInt resultX;
    BigInt resultY;
    BigInt top;
    BigInt bottom;
    BigInt otherX;
    BigInt l;
    BigInt x3;

    while (e > 0)
    {
        cout << e << endl;
        if (true)
        {
            if (!have)
            {
                have = true;
                resultX = qX;
                resultY = qY;
            }
            else if (qX == resultX)
            {
                top = BigInt(3) * qX * qX;
                bottom = BigInt(2) * qY;
                otherX = qX;
                l = (top * inverseMod(&bottom, &P)) % P;
                x3 = (l * l - qX - otherX) % P;
                resultX = resultX + x3;
                resultY = resultY + (l * (qX - x3) - qY) % P;
            }
            else
            {
                top = resultY - qY;
                bottom = resultX - qX;
                otherX = resultX;
                l = (top * inverseMod(&bottom, &P)) % P;
                x3 = (l * l - qX - otherX) % P;
                resultX = x3;
                resultY = (l * (qX - x3) - qY) % P;
            }
        }
        //top, bottom, otherX = BigInt(3) * qX * qX, BigInt(2) * qY, qX;
        top = BigInt(3) * qX * qX;
        bottom = BigInt(2) * qY;
        otherX = qX;


        l = (top * inverseMod(&bottom, &P)) % P;
        x3 = (l * l - qX - otherX) % P;
        e = e >> 1;
        qX = x3;
        qY = (l * (qX - x3) - qY) % P;
        //e, qX, qY = e >> 1, x3, (l * (qX - x3) - qY) % P;
    }
    cout << "04 " << resultX << " " << resultY << endl;
    return 0;
}