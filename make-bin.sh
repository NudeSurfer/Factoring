
##/opt/intel/bin/icc -Wall -g -march=skylake-avx512 -o AVXtest Factorial_AVX512.c -lm
/opt/intel/bin/icc -Wall -O2 -march=skylake-avx512 -o AVXtestFinal Factorial_AVX512_final.c -lm
