#! /usr/bin/env python3
g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
g_y = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
#g_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
#secp256k1
# print(g_x)
# print(g_y)
# print(g_order)
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
#print("p:" , p)
# secret key 
priv = 83020463237742095447187601505567755285685578327720972455401612915394172527322
print("priv: ", priv)

def inverse_mod(a):
    if a < 0 or a >= p: 
        a = a % p
    c, d, uc, vc, ud, vd = a, p, 1, 0, 0, 1
    while c:
        #q, c, d = divmod(d, c) + (c,)
        q, c, d = d // c, d % c, c
        # print(q)
        # print(c)
        # print(d)
        uc, vc, ud, vd = ud - q*uc, vd - q*vc, uc, vc
    if ud > 0: 
        return ud
    return ud + p

q_x, q_y  = g_x, g_y
have = False
e = priv
while e:
    print(e)
    if True: 
        if not have: 
            # Выполняется только первый раз после выполнения условия
            have = True
            result_x, result_y = q_x, q_y 
        elif q_x == result_x:
            top, bottom, other_x = 3 * q_x* q_x, 2 * q_y, q_x
            l = (top * inverse_mod(bottom)) % p
            x3 = (l * l - q_x- other_x) % p
            result_x += x3
            result_y += (l * (q_x - x3) - q_y) % p
        else:
            top, bottom, other_x = result_y - q_y, result_x - q_x, result_x
            l = (top * inverse_mod(bottom)) % p
            x3 = (l * l - q_x - other_x) % p
            result_x = x3
            result_y = (l * (q_x - x3) - q_y) % p
    top, bottom, other_x = 3 * q_x * q_x, 2 * q_y, q_x
    l = (top * inverse_mod(bottom)) % p
    x3 = (l * l - q_x - other_x) % p
    e, q_x, q_y = e >> 1, x3, (l * (q_x - x3) - q_y) % p
#print (priv)
print(result_x, result_y)
print ('  privkey:    %x\n   pubkey: %s' % (priv, "04 %x %x" % (result_x, result_y)) )