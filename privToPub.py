#! /usr/bin/env python
g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
g_y = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
g_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
#secp256k1
p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
# secret key 
priv =  0xb78beac0be5748a3b18f10babf69cfa8e18461fab68dfefbdc90780d2909a2da

def inverse_mod(a):
    if a < 0 or a >= p: 
        a = a % p
    c, d, uc, vc, ud, vd = a, p, 1, 0, 0, 1
    while c:
        q, c, d = divmod(d, c) + (c,)
        uc, vc, ud, vd = ud - q*uc, vd - q*vc, uc, vc
    if ud > 0: 
        return ud
    return ud + p

def mul(point_x, point_y, e):
    q_x, q_y  = point_x, point_y
    have = False
    while e:
        if e&1: 
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
    return "04 %x %x" % (result_x, result_y)
print ('  privkey:    %x\n   pubkey: %s' % (priv, mul(g_x, g_y, priv)) )