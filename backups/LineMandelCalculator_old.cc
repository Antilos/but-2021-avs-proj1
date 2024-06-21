/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	mandelMask = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	for(size_t i = 0; i < height * width; i++){
		data[i] = limit;
		mandelMask[i] = 0;
	}
	imags = (float *)(_mm_malloc(height * sizeof(float), 64));
	reals = (float *)(_mm_malloc(width * sizeof(float), 64));
	zImags = (float *)(_mm_malloc(width * sizeof(float), 64));
	zReals = (float *)(_mm_malloc(width * sizeof(float), 64));

	for(size_t i = 0, j = 0; i < height || j < width; i++, j++){
		*(imags + i) = y_start + i * dy;
		*(reals + j) = x_start + j * dx;
	}

	maskSum = 0;
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
	_mm_free(mandelMask);
	mandelMask = NULL;
	_mm_free(zImags);
	mandelMask = NULL;
	_mm_free(zReals);
	mandelMask = NULL;
}

// template <typename T>
// static inline int mandelbrot(T real, T imag, int limit)
// {
// 	T zReal = real;
// 	T zImag = imag;

// 	for (int i = 0; i < limit; ++i)
// 	{
// 		T r2 = zReal * zReal;
// 		T i2 = zImag * zImag;

// 		if (r2 + i2 > 4.0f)
// 			return i;

// 		zImag = 2.0f * zReal * zImag + imag;
// 		zReal = r2 - i2 + real;
// 	}
// 	return limit;
// }

int * LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int *pdata = data;
	float *pzImags = zImags;
	float *pzReals = zReals;
	int *pmandelMask = mandelMask;
	for (int i = 0; i < height; i++)
	{
		for(size_t itmp = 0; itmp < width; i++){
			*(pzImags + itmp) = *(imags + i);
			*(pzReals + itmp) = *(reals + itmp);
		}
		// maskSum = 0;
		for (int k = 0; !(maskSum >= width) || k < limit; ++k)
		{
			#pragma omp simd simdlen(64) aligned(pdata, pzImags, pzReals, pmandelMask:64) reduction(+:maskSum)
			for (int j = 0; j < width; j++)
			{
				// //Current position
				// float y = y_start + i * dy; // current imaginary value
				// float x = x_start + j * dx; // current real value

				//Current value
				pzImags[width * i + j] = (k == 0 ? y : pzImags[width * i + j]);
				pzReals[width * i + j] = (k == 0 ? x : pzReals[width * i + j]);
				maskSum = ((k == 0 && j == 0) ? 0 : maskSum); //watch out probably wont paralelize well

				float r2 = pzReals[width * i + j] * pzReals[width * i + j];
				float i2 = pzImags[width * i + j] * pzImags[width * i + j];

				//check and set current value
				//pdata[width * i + j] = (k + 1 == limit && !mandelMask[width * i + j]) ? limit : ((!mandelMask[width * i + j] && (r2 + i2 > 4.0f)) ? k : pdata[width * i + j]); //check wether we've reached limit
				pdata[width * i + j] = (!pmandelMask[width * i + j] && (r2 + i2 > 4.0f)) ? k : pdata[width * i + j]; //set value to current iteration
				maskSum += (!pmandelMask[width * i + j] && (r2 + i2 > 4.0f)) ? 1 : 0;
				pmandelMask[width * i + j] = (pmandelMask[width * i + j] || (r2 + i2 > 4.0f)) ? 1 : 0;

				//prepare for next iteration of mandelbrot
				pzImags[width * i + j] =  2.0f * pzReals[width * i + j] * pzImags[width * i + j] + y;
				pzReals[width * i + j] = r2 - i2 + x;

			}
		}
	}
	return data;
}
