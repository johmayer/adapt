#Step 1b (case for user-defined multiplier)
########################################################
import numpy as np
i = 5
multiplier = np.load(f"S_{i}.npy") 
def mulX(a,b):
    result = multiplier[a + 128,b+128]
    return result
   
use_evo = False

class my_accurate_mult(object):
    def mul(self, a, b):
        return mulX(a,b)

#Step 2
#########################################################   
if use_evo:
    # for EvoApprox multiplier
    axx_mult = mul8s_1L2H # change name appropriately
else:
    # for user-defined multiplier
    axx_mult = my_accurate_mult()
    
    
#Step 3
#########################################################
use_signed_conversion = False
###### DO NOT CHANGE ######

nbits = 8

if use_signed_conversion:
    #for the case of signed conversion
    def u2s(v): # 16b unsigned to 16b signed
        if v & 32768:
            return v - 65536
        return v
else:  
    #for the case of no conversion
    def u2s(v): 
        return v


#Step 4
########################################################
mult_name = 'S_5'


#Step 5
########################################################
with open(mult_name + '.h', 'w') as filehandle: 
    bits = int(pow(2,nbits))
    lut_size_str = str(bits)

    filehandle.write('#include <stdint.h>\n\n')
    filehandle.write('const int' + str(2*nbits) + '_t lut [' + lut_size_str + '][' + lut_size_str +'] = {')       
    
    for i in range (0,bits//2):
        filehandle.write('{')
        for j in range (0,bits//2):
            x = u2s(axx_mult.mul(i,j))
            filehandle.write('%s' % x)
            filehandle.write(', ')  
        for j in range (bits//2,bits):
            x = u2s(axx_mult.mul(i,(bits-j)*-1))
            filehandle.write('%s' % x)
            if j!=bits-1:
                filehandle.write(', ') 
        filehandle.write('},')  
        filehandle.write('\n')
        
    for i in range (bits//2,bits):
        filehandle.write('{')
        for j in range (0,bits//2):
            x = u2s(axx_mult.mul((bits-i)*-1,j))        
            filehandle.write('%s' % x)
            filehandle.write(', ')  
        for j in range (bits//2,bits):
            x = u2s(axx_mult.mul((bits-i)*-1,(bits-j)*-1))
            filehandle.write('%s' % x)
            if j!=bits-1:
                filehandle.write(', ')
        if(i!=bits-1):        
            filehandle.write('},')
            filehandle.write('\n')
    filehandle.write('}};\n')        