from fractions import Fraction

def testFactoriaFactor(n, factor):
   residue = 2
   for i in range(3,n+1):
       residue = residue *i
       residue = residue % factor
   return residue

def fma(a, b, c):
    # Return x*y + z with only a single rounding.
   return float(Fraction(a)*Fraction(b) + Fraction(c))
   
def fms(a, b, c):
    # Return x*y + z with only a single rounding.
   return float(Fraction(a)*Fraction(b) - Fraction(c))
   
def two_prod(a, b):
	# double-double = double * double 
   high = float(a)*float(b)
   low = fma(float(a), float(b), -high)
   return low, high

def quick_two_sum(a, b):
   high = float (a) + float(b)
   low = float(b) - (high - float(a))
   return low, high

def	dd_mul_d(a_high, a_low, b):
#double-double * double 

   p_low, p_high,  = two_prod(float(a_high), float(b)) 
   p_low += float(a_low) * float(b)
   p_low, p_high = quick_two_sum(p_high, p_low)
 
def FMAFactoriaFactor(n, p):

   remainder = 2.0
   pd = float(p)
   reciprocal = 1/pd
   rounding_constant = 6755399441055744.0
   
   for i in range (3, n+1):
      ab = float(i)* remainder 
	  # ab can be > 2^53 so this is approximation of the product i*p
	  # because ab is max 10^7 * p error is propably around 10^7 and << p (p can be 10^13) 
     # print("Round i: " + str(i))
     # print("ab: " + str(ab))
      quotient = fma(ab, reciprocal, rounding_constant)
      quotient -= rounding_constant
	  
     # print("quotient: " + str(quotient)) 
      high = pd *quotient  
    #  print ("high:" + str(high))
      low =	fms(pd, quotient,high)
   #   print ("low:" + str(low))
      # here high and low are accurate product of the p* quotient
      # here high and low are accurate product of the p* quotient
	  # because of the FMA producr aka decker product
	  # but we need to remember that quitient was calculated using approximate reminder *i
	  
      remainder =fms(remainder, float(i), high) - low
  #    print ("remainder:" + str(remainder))
      if (remainder < 0.0 ):
         remainder += pd
 #     print ("remainder:" + str(remainder))
   return int(remainder)
	

#1009 | 1007!-1
#1009 | 1008!+1
#100003 | 100001!-1	
#print("now testing FMA")

#print(FMAFactoriaFactor(1007, 1009))
#print(FMAFactoriaFactor(1008, 1009)) 
#print(FMAFactoriaFactor(100001, 100003)) 
#print(FMAFactoriaFactor(1973933, 100059931987))
filename = input("Give file name to test factors:")
count = 0
with open(filename, 'r') as f:
   for line in f:
      factor, end_part1 = line.split("|")
      n, end_part2 = end_part1.split("!")
      modulo = testFactoriaFactor(int(n), int(factor))
      if modulo !=1 and end_part2.strip() == "-1":
          print(factor + " doesnt divide " + n.strip() + end_part2)
      if modulo !=(int(factor)-1) and end_part2.strip() == "+1":
          print(factor + " doesnt divide " + n.strip() + end_part2)

      count += 1
   f.close()
   print(str(count) +" factors tested")

  
