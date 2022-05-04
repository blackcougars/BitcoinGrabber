#include "Kernel.h"
#include "BigInt.h"


__device__ BigInt inverseMod (BigInt* a, BigInt* p)
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


__global__ void kernel(string* arrayData, long int* countData, string* progress)
{   
    // Ядро проекта
    unsigned long int warp = 1'000'000; // Размер сдвига каждого потока
    BigInt startPriv = BigInt(*progress) + (threadIdx.x * warp) - warp;     // Начальный ключ

    BigInt jumpWarp = startPriv + warp; // Число после 
    
    //atomicAdd(countCheckedPtr, 1); // 
    bool running = true;
    BigInt currentPriv = startPriv;
    BigInt endKeys = BigInt("115792089237316195423570985008687907853269984665640564039457584007913129639936"); // End bitcoin valid private keys
    
    // secp256k1
    BigInt gX = BigInt("55066263022277343669578718895168534326250603453777594175500187360389116729240");
    BigInt gY = BigInt("32670510020758816978083085130507043184471273380659243275938904335757337482424");
    BigInt gOrder = BigInt("115792089237316195423570985008687907852837564279074904382605163141518161494337");
    BigInt P = BigInt("115792089237316195423570985008687907853269984665640564039457584007908834671663");
    
    while (running)
    {
        while(currentPriv <= jumpWarp)
        {
            // Priv key to pub and check in db
            // Private key to public key
            BigInt pubKey;

            BigInt qX = gX;
            BigInt qY = gY;
            bool have = false;
            BigInt e = currentPriv;
            
            BigInt resultX;
            BigInt resultY;

            BigInt top;
            BigInt bottom;
            BigInt otherX;
            BigInt l;
            BigInt x3;

            while (e > 0)
            {
                if (e&1)
                {
                    if (!have)
                    {
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
                        resultX += x3;
                        resultY += (l * (qX - x3) - qY) % P;
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
                top, bottom, otherX = BigInt(3) * qX * qX, BigInt(2) * qY, qX;
                l = (top * inverseMod(&bottom, &P)) % P;
                x3 = (l * l - qX - otherX) % P;
                e, qX, qY = e >> 1, x3, (l * (qX - x3) - qY) % P;
            }

            // check in db
            int i = 0;
            string pubKeyStr = pubKey.getValue();
            while(i <= *countData)
            {
                if (*(arrayData + i) == pubKeyStr)
                    // Winner 
                    ;
                i += 1;
            }
            currentPriv = currentPriv + 1; // increment (fasted)

        }
        jumpWarp = jumpWarp + (1024 * warp);
        if (currentPriv >= endKeys)
            running = false;
    }
}