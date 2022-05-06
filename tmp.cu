#include "BigInt.h"
#include <iostream>

using namespace std;

__host__ BigInt inverseMod (BigInt* a, const BigInt* const p)
{
    if ( *a < 0 || *a >= *p) 
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
        BigInt tmp = c;
        c = d % c;
        d = tmp;

        BigInt tmpUc = uc;
        BigInt tmpVc = vc;
        BigInt tmpUd = ud;
        BigInt tmpVd = vd;

        uc = tmpUd - (q*tmpUc);
        vc = tmpVd - (q*tmpVc);
        ud = tmpUc;
        vd = tmpVc;
    }
    if (ud > 0) 
        return ud;
    
    return ud + *p;
}


__host__ int main()
{   
    
    //BigInt c ("128319238");
    //c = c >> 1;
    //c = c >> 1;
    

    //cout << c & 1;
    // if (c & 1)
    //     cout << "1" << endl;
    // else
    //     cout << "0" << endl;
    //return 0;

    // secp256k1
    BigInt gX = BigInt("55066263022277343669578718895168534326250603453777594175500187360389116729240");
    BigInt gY = BigInt("32670510020758816978083085130507043184471273380659243275938904335757337482424");
    BigInt gOrder = BigInt("115792089237316195423570985008687907852837564279074904382605163141518161494337");
    const BigInt P = BigInt("115792089237316195423570985008687907853269984665640564039457584007908834671663");
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
        //cout << e << endl;
        if (e & 1)
        {
            if (!have)
            {
                have = true;
                resultX = qX;
                resultY = qY;
                //cout << resultX << endl;
            }
            // else if (qX == resultX)
            // {
            //     top = BigInt(3) * qX * qX;
            //     bottom = BigInt(2) * qY;
            //     otherX = qX;
            //     //cout <<inverseMod(&bottom, &P) << endl;
            //     l = (top * inverseMod(&bottom, &P)) % P;
            //     x3 = (l * l - qX - otherX) % P;
            //     resultX = resultX + x3;
            //     resultY = resultY +  (l * (qX - x3) - qY) % P;
            // }
            else
            {
                top = resultY - qY;
                //cout << top << endl;
                bottom = resultX - qX;
                otherX = resultX;
                //cout <<inverseMod(&bottom, &P) << endl;
                l = (top * inverseMod(&bottom, &P)) % P;
                x3 = (l * l - qX - otherX) % P;
                //cout << x3 << endl;

                resultX = x3;
                resultY = (l * (qX - x3) - qY) % P;
            }
        }
    

        top = BigInt(3) * qX * qX;
        bottom = BigInt(2) * qY;
        otherX = qX;
        l = (top * inverseMod(&bottom, &P)) % P;
        x3 = (l * l - qX - otherX) % P;
        cout << e << endl;
        e = e >> 1;
        BigInt tmp = qX;
        qX = x3;
        qY = (l * (tmp - x3) - qY) % P;
        //e, qX, qY = e >> 1, x3, (l * (qX - x3) - qY) % P;

        //cout << resultX << endl;
    }
    ////cout << e << endl;
    cout << resultX << endl;
    //cout << "04 " << resultX << " " << resultY << endl;
    return 0;
}