#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "iacaMarks.h"
#include <inttypes.h>
#include <sys/time.h>

#define APP_BUFLEN 32
#define FACTORS_FILENAME_DEFAULT "fpfactors.txt"

static const double rounding_constant = 6755399441055744.0; // 2^51 + 2^52 magic number

double stoptime(void) {
	struct timeval t;
	gettimeofday(&t,NULL);
	return (double) t.tv_sec + t.tv_usec/1000000.0;
}

void report_factor(uint64_t n, int64_t c, uint64_t p)
{
	/*
		FILE *file;
		
		if ((file = fopen(FACTORS_FILENAME_DEFAULT,"a")) != NULL)
		{
		fprintf(file,"%"PRIu64" | %"PRIu64"!%+"PRId64"\n",p,n,c);
		fclose(file);
	}*/
	printf("%"PRIu64" | %"PRIu64"!%+"PRId64"\n",p,n,c);
}


long FindPrimesUpTo(uint64_t lim, uint64_t **prime_table){
	
	uint64_t i,j, limit, count;
	uint64_t *table;
	limit = lim +1;
	// This takes quite a lot memory, but ok just for testing. I have 128GB.
	
	char *is_composite = calloc(limit, sizeof(char));
	
	uint64_t squareroot_limit = sqrt(limit);
	for (i = 2; i <= squareroot_limit; i++){
		if (!is_composite[i]){
			for (j = i * i; j < limit; j += i){
				is_composite[j] = 1;
			}
		}
	}
	
	count = 0;
	for (i=2; i < limit; i++){
		if (!is_composite[i]) count++;
	}
	table = aligned_alloc(64, count * sizeof(uint64_t));
	
	count = 0;
	for (i=2; i < limit; i++){
		if (!is_composite[i]){
			table[count]= i;
			count++;
		}
	}
	*prime_table = table;
	free(is_composite);
	return count;
}
// for debugging purposes
void print512_pd(__m512d in) {
	double v[8];
	_mm512_storeu_pd((__m512d*)v, in);
	printf("v8_pd: %f %f %f %f %f %f %f %f\n",
	v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}
// for debugging purposes
void print512_u64(__m512i in) {
	uint64_t v[8];
	_mm512_storeu_si512((__m512i*)v, in);
	printf("v8_u64: %lld %lld %lld %lld %lld %lld %lld %lld\n",
	v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

uint64_t factorial_naive(uint64_t const nmin, uint64_t const nmax, const uint64_t *restrict P)
{
	uint64_t n, i, residue;
	
	for (i = 0; i < APP_BUFLEN; i++){
		residue = 2;
		for (n=3; n <= nmax; n++){
			residue *=  n;
			residue %= P[i];
			// Lets check if we found factor
			if (nmin <= n){
				if( residue == 1){
					report_factor(n, -1, P[i]);
				}
				if(residue == P[i]- 1){
					report_factor(n, 1, P[i]);
				}
			}
		}
	}
	return EXIT_SUCCESS;
}

int64_t factorial_FMA(uint64_t const nmin, uint64_t const nmax, const uint64_t *restrict P)
{
	uint64_t n, i;
	double prime_double, prime_double_reciprocal, quotient, residue;
	double nr, n_double, prime_times_quotient_high, prime_times_quotient_low;
	
	for (i = 0; i < APP_BUFLEN; i++){
		residue = 2.0;
		prime_double = (double)P[i];
		prime_double_reciprocal = 1.0 / prime_double;
		n_double = 3.0;
		for (n=3; n <= nmax; n++){
			
			// this is not accurate when product is > 2^53
			nr =  n_double * residue;
			// next we use FMA version of the Decker product (1971)
			// information can be found e.g. Handbook of Floating-point arithmetic,
			// FMA version of TwoProduct algorithm, double machine precision arithmetic,
			// and e.g 2.5 Double length multiplication from Multiple precision floating point
			// arithmetic on SIMD processors by Joris Van Der Hoeven
			// fma has been part of C standard since 99
			quotient = fma(nr, prime_double_reciprocal, rounding_constant);
			
			quotient -= rounding_constant;
			
			// this is double * double -> double-double product, accurate up to 106 bits.
			// low has value only when high > 2^53
			prime_times_quotient_high= prime_double * quotient;
			prime_times_quotient_low = fma(prime_double, quotient, -prime_times_quotient_high);
			
			// because our nr is not accurate ( >2^53) lets calculate new residue using
			//  accurate n * original residue
			
			residue = fma(residue, n, -prime_times_quotient_high) - prime_times_quotient_low;
			// if residue is smaller than zero because of rounding of the quotient lets correct that
			//printf("residue %f\n", residue);
			if (residue < 0.0) residue += prime_double;
			n_double += 1.0;
			
			// Lets check if we found factor
			if (nmin <= n){
				if( residue == 1.0){
					report_factor(n, -1, P[i]);
				}
				if(residue == prime_double - 1.0){
					report_factor(n, 1, P[i]);
				}
			}
		}
	}
	return EXIT_SUCCESS;
}


uint64_t factorial_AVX512_unrolled_four(uint64_t const nmin, uint64_t const nmax, const uint64_t *restrict P)
{
	// we are trying to find a factor for a factorial numbers : n! +-1
	//nmin is minimun n we want to report and nmax is maximum. P is table of primes
	// we process 32 primes in one loop.
	// naive version of the algorithm is int he fucntion factorial_naive
	// and simple version of the FMA based approach in the function factorial_simpleFMA
	
	const double one_table[8] __attribute__ ((aligned(64))) ={1.0, 1.0, 1.0,1.0,1.0,1.0,1.0,1.0};
	
	
	uint64_t n;
	
	__m512d zero, rounding_const, one, n_double;
	
	__m512i prime1, prime2, prime3, prime4;
	
	__m512d residue1, residue2, residue3, residue4;
	__m512d prime_double_reciprocal1, prime_double_reciprocal2, prime_double_reciprocal3, prime_double_reciprocal4;
	__m512d quotient1, quotient2, quotient3, quotient4;
	__m512d prime_times_quotient_high1, prime_times_quotient_high2, prime_times_quotient_high3, prime_times_quotient_high4;
	__m512d prime_times_quotient_low1, prime_times_quotient_low2, prime_times_quotient_low3, prime_times_quotient_low4;
	__m512d nr1, nr2, nr3, nr4;
	__m512d prime_double1, prime_double2, prime_double3, prime_double4;
    __m512d prime_minus_one1, prime_minus_one2, prime_minus_one3, prime_minus_one4;
	
	__mmask8 negative_reminder_mask1, negative_reminder_mask2, negative_reminder_mask3, negative_reminder_mask4;
	__mmask8 found_factor_mask11, found_factor_mask12, found_factor_mask13, found_factor_mask14;
	__mmask8 found_factor_mask21, found_factor_mask22, found_factor_mask23, found_factor_mask24;
	
	// load data and initialize cariables for loop
	rounding_const = _mm512_set1_pd(rounding_constant);
	one = _mm512_load_pd(one_table);
	zero = _mm512_setzero_pd ();
	
	// load primes used to sieve
	prime1 = _mm512_load_epi64((__m512i *) &P[0]);
	prime2 = _mm512_load_epi64((__m512i *) &P[8]);
	prime3 = _mm512_load_epi64((__m512i *) &P[16]);
	prime4 = _mm512_load_epi64((__m512i *) &P[24]);
	
	// convert primes to double
	prime_double1 = _mm512_cvtepi64_pd (prime1); // vcvtqq2pd
	prime_double2 = _mm512_cvtepi64_pd (prime2); // vcvtqq2pd
	prime_double3 = _mm512_cvtepi64_pd (prime3); // vcvtqq2pd
	prime_double4 = _mm512_cvtepi64_pd (prime4); // vcvtqq2pd
	
	// calculates 1.0/ prime
	prime_double_reciprocal1 = _mm512_div_pd(one, prime_double1);
	prime_double_reciprocal2 = _mm512_div_pd(one, prime_double2);
	prime_double_reciprocal3 = _mm512_div_pd(one, prime_double3);
	prime_double_reciprocal4 = _mm512_div_pd(one, prime_double4);
	
	// for comparison if we have found factors for n!+1
	prime_minus_one1 = _mm512_sub_pd(prime_double1, one);
	prime_minus_one2 = _mm512_sub_pd(prime_double2, one);
	prime_minus_one3 = _mm512_sub_pd(prime_double3, one);
	prime_minus_one4 = _mm512_sub_pd(prime_double4, one);
	
	// residue init
	residue1 =  _mm512_set1_pd(2.0);
	residue2 =  _mm512_set1_pd(2.0);
	residue3 =  _mm512_set1_pd(2.0);
	residue4 =  _mm512_set1_pd(2.0);
	
	// double counter init
	n_double = _mm512_set1_pd(3.0);
	
	// main loop starts here. typical value for nmax can be 5,000,000 -> 10,000,000
	
	for (n=3; n<=nmax; n++) // main loop
	{
		
		// timings for instructions:
		// _mm512_load_epi64 = vmovdqa64 : L 1, T 0.5
		// _mm512_load_pd = vmovapd : L 1, T 0.5
		// _mm512_set1_pd
		// _mm512_div_pd = vdivpd : L 23, T 16
		// _mm512_cvtepi64_pd = vcvtqq2pd : L 4, T 0,5
		
		// _mm512_mul_pd = vmulpd :  L 4, T 0.5
		// _mm512_fmadd_pd = vfmadd132pd, vfmadd213pd, vfmadd231pd :  L 4, T 0.5
		// _mm512_fmsub_pd = vfmsub132pd, vfmsub213pd, vfmsub231pd : L 4, T 0.5
		// _mm512_sub_pd = vsubpd : L 4, T 0.5
		// _mm512_cmplt_pd_mask = vcmppd : L ?, Y 1
		// _mm512_mask_add_pd = vaddpd :  L 4, T 0.5
		// _mm512_cmpeq_pd_mask = vcmppd L ?, Y 1
		// _mm512_kor = korw L 1, T 1
		
		// nr = residue *  n
		nr1 = _mm512_mul_pd (residue1, n_double);
		nr2 = _mm512_mul_pd (residue2, n_double);
		nr3 = _mm512_mul_pd (residue3, n_double);
		nr4 = _mm512_mul_pd (residue4, n_double);
		
		// quotient = nr * 1.0/ prime_double + rounding_constant
		quotient1 = _mm512_fmadd_pd(nr1, prime_double_reciprocal1, rounding_const);
		quotient2 = _mm512_fmadd_pd(nr2, prime_double_reciprocal2, rounding_const);
		quotient3 = _mm512_fmadd_pd(nr3, prime_double_reciprocal3, rounding_const);
		quotient4 = _mm512_fmadd_pd(nr4, prime_double_reciprocal4, rounding_const);
		
		// quotient -= rounding_constant, now quotient is rounded to integer
		// countient should be at maximum nmax (10,000,000)
		quotient1 = _mm512_sub_pd(quotient1, rounding_const);
		quotient2 = _mm512_sub_pd(quotient2, rounding_const);
		quotient3 = _mm512_sub_pd(quotient3, rounding_const);
		quotient4 = _mm512_sub_pd(quotient4, rounding_const);
		
		// now we calculate high and low for prime * quotient using decker product (FMA).
		// quotient is calculated using approximation but this is accurate for given quotient
		prime_times_quotient_high1 = _mm512_mul_pd(quotient1, prime_double1);
		prime_times_quotient_high2 = _mm512_mul_pd(quotient2, prime_double2);
		prime_times_quotient_high3 = _mm512_mul_pd(quotient3, prime_double3);
		prime_times_quotient_high4 = _mm512_mul_pd(quotient4, prime_double4);
		
		
		prime_times_quotient_low1 = _mm512_fmsub_pd(quotient1, prime_double1, prime_times_quotient_high1);
		prime_times_quotient_low2 = _mm512_fmsub_pd(quotient2, prime_double2, prime_times_quotient_high2);
		prime_times_quotient_low3 = _mm512_fmsub_pd(quotient3, prime_double3, prime_times_quotient_high3);
		prime_times_quotient_low4 = _mm512_fmsub_pd(quotient4, prime_double4, prime_times_quotient_high4);
		
		// now we calculate new reminder using decker product and using original values
		// we subtract above calculated prime * quotient (quotient is aproximation)
		
		residue1 = _mm512_fmsub_pd(residue1, n_double, prime_times_quotient_high1);
		residue2 = _mm512_fmsub_pd(residue2, n_double, prime_times_quotient_high2);
		residue3 = _mm512_fmsub_pd(residue3, n_double, prime_times_quotient_high3);
		residue4 = _mm512_fmsub_pd(residue4, n_double, prime_times_quotient_high4);
		
		residue1 = _mm512_sub_pd(residue1, prime_times_quotient_low1);
		residue2 = _mm512_sub_pd(residue2, prime_times_quotient_low2);
		residue3 = _mm512_sub_pd(residue3, prime_times_quotient_low3);
		residue4 = _mm512_sub_pd(residue4, prime_times_quotient_low4);
		
		// lets check if reminder < 0
		negative_reminder_mask1 = _mm512_cmplt_pd_mask(residue1,zero);
		negative_reminder_mask2 = _mm512_cmplt_pd_mask(residue2,zero);
		negative_reminder_mask3 = _mm512_cmplt_pd_mask(residue3,zero);
		negative_reminder_mask4 = _mm512_cmplt_pd_mask(residue4,zero);
		
		// we and prime back to reminder using mask if it was < 0
		residue1 = _mm512_mask_add_pd(residue1, negative_reminder_mask1, residue1, prime_double1);
		residue2 = _mm512_mask_add_pd(residue2, negative_reminder_mask2, residue2, prime_double2);
		residue3 = _mm512_mask_add_pd(residue3, negative_reminder_mask3, residue3, prime_double3);
		residue4 = _mm512_mask_add_pd(residue4, negative_reminder_mask4, residue4, prime_double4);
		
		n_double = _mm512_add_pd(n_double,one);
		
		// if we are below nmin then we continue next iteration, we
		if (n < nmin) continue;
		
		// Lets check if we found any factors, residue 1 == n!-1
		found_factor_mask11 = _mm512_cmpeq_pd_mask(one, residue1);
		found_factor_mask12 = _mm512_cmpeq_pd_mask(one, residue2);
		found_factor_mask13 = _mm512_cmpeq_pd_mask(one, residue3);
		found_factor_mask14 = _mm512_cmpeq_pd_mask(one, residue4);
		
		// residue prime -1  == n!+1
		found_factor_mask21 = _mm512_cmpeq_pd_mask(prime_minus_one1, residue1);
		found_factor_mask22 = _mm512_cmpeq_pd_mask(prime_minus_one2, residue2);
		found_factor_mask23 = _mm512_cmpeq_pd_mask(prime_minus_one3, residue3);
		found_factor_mask24 = _mm512_cmpeq_pd_mask(prime_minus_one4, residue4);
		
		/*	// lets combine results
			found_factor_mask11 = _mm512_kor(found_factor_mask11, found_factor_mask21);
			found_factor_mask12 = _mm512_kor(found_factor_mask12, found_factor_mask22);
			found_factor_mask13 = _mm512_kor(found_factor_mask13, found_factor_mask23);
		found_factor_mask14 = _mm512_kor(found_factor_mask14, found_factor_mask24); */
		
		
		if (found_factor_mask12 | found_factor_mask11 | found_factor_mask13 | found_factor_mask14 |
		found_factor_mask21 | found_factor_mask22 | found_factor_mask23|found_factor_mask24)
		{ // we find factor very rarely
			
			double *residual_list1 = (double *) &residue1;
			double *residual_list2 = (double *) &residue2;
			double *residual_list3 = (double *) &residue3;
			double *residual_list4 = (double *) &residue4;
			
			double *prime_list1 = (double *) &prime_double1;
			double *prime_list2 = (double *) &prime_double2;
			double *prime_list3 = (double *) &prime_double3;
			double *prime_list4 = (double *) &prime_double4;
			
			
			
			for (int i=0; i <8; i++){
				if( residual_list1[i] == 1.0)
				{
					report_factor((uint64_t) n, -1, (uint64_t) prime_list1[i]);
				}
				if( residual_list2[i] == 1.0)
				{
					report_factor((uint64_t) n, -1, (uint64_t) prime_list2[i]);
				}
				if( residual_list3[i] == 1.0)
				{
					report_factor((uint64_t) n, -1, (uint64_t) prime_list3[i]);
				}
				if( residual_list4[i] == 1.0)
				{
					report_factor((uint64_t) n, -1, (uint64_t) prime_list4[i]);
				}
				
				if(residual_list1[i] == (prime_list1[i] - 1.0))
				{
					report_factor((uint64_t) n, 1, (uint64_t) prime_list1[i]);
				}
				if(residual_list2[i] == (prime_list2[i] - 1.0))
				{
					report_factor((uint64_t) n, 1, (uint64_t) prime_list2[i]);
				}
				if(residual_list3[i] == (prime_list3[i] - 1.0))
				{
					report_factor((uint64_t) n, 1, (uint64_t) prime_list3[i]);
				}
				if(residual_list4[i] == (prime_list4[i] - 1.0))
				{
					report_factor((uint64_t) n, 1, (uint64_t) prime_list4[i]);
				}
			}
			// return found_factor_mask1;
		}
		
	}
	
	return EXIT_SUCCESS;
}

main ()
{
	// there are 78,498 primes below 1,000,000, 9,592 below 100,000,
	// 664,579 below 10,000,000 and  664,584 below 10,000,140
	//10000141  10000169
	// sieve start and end limits have been selected in such way that they are around 10,000,000
	// and 100,000,000 but that there is 32 multiple of primes between them
	uint64_t i=0,j , sieve_limit = 10000140, sieve_min = 9000000, rounds =500;
	uint64_t *prime_table, number_of_primes;
	double start_time, end_time;
	
	number_of_primes = FindPrimesUpTo(sieve_limit, &prime_table);
	printf("Number of primes used for sieving: %"PRIu64"\n",number_of_primes);
	
	// benchmark AVX 512
	for (i=0; prime_table[i] < sieve_min; i +=APP_BUFLEN)
	start_time = stoptime();
	
	for (j=0  ; j < rounds; j++){
		factorial_AVX512_unrolled_four(1000, 4000000, &prime_table[i]);
		i +=(APP_BUFLEN);
	}
	end_time = stoptime();
	printf("time used(unrolled version):\t %.3f seconds, primes per second: \t %.3f\n", end_time -start_time, rounds * APP_BUFLEN/ (end_time -start_time) );
	
	// benchmark FMA
	for (i=0; prime_table[i] < sieve_min; i +=APP_BUFLEN)
	start_time = stoptime();
	
	for (j=0  ; j < rounds / 50; j++){
		factorial_FMA(1000, 4000000, &prime_table[i]);
		i +=(APP_BUFLEN);
	}
	end_time = stoptime();
	printf("time used(FMA version):\t %.3f seconds, primes per second: \t %.3f\n", end_time -start_time, (rounds /50) * APP_BUFLEN/ (end_time -start_time) );
	
	// benchmark naive method
	for (i=0; prime_table[i] < sieve_min; i +=APP_BUFLEN)
	start_time = stoptime();
	
	for (j=0  ; j < rounds / 50; j++){
		factorial_naive(1000, 4000000, &prime_table[i]);
		i +=(APP_BUFLEN);
	}
	end_time = stoptime();
	printf("time used(naive version):\t %.3f seconds, primes per second: \t %.3f\n", end_time -start_time, (rounds/50) * APP_BUFLEN/ (end_time -start_time) );
	
	
	return EXIT_SUCCESS;
}
