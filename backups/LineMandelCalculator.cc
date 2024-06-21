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
	buff.data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	buff.mask = (int *)(_mm_malloc(width * sizeof(int), 64));
	for(size_t i = 0; i < height * width; i++){
		*((buff.data) + i) = limit;
	}
	buff.imags = (float *)(_mm_malloc(height * sizeof(float), 64));
	buff.reals = (float *)(_mm_malloc(width * sizeof(float), 64));
	buff.zImags = (float *)(_mm_malloc(width * sizeof(float), 64));
	buff.zReals = (float *)(_mm_malloc(width * sizeof(float), 64));

	for(size_t i = 0; i < height; i++){
		*((buff.imags) + i) = y_start + i * dy;
	}
	for(size_t j = 0; j < width; j++){
		*((buff.reals) + j) = x_start + j * dx;
		*((buff.mask) + j) = 0;
	}

	// maskSum = 0;
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(buff.data);
	buff.data = NULL;
	_mm_free(buff.mask);
	buff.mask = NULL;
	_mm_free(buff.imags);
	buff.imags = NULL;
	_mm_free(buff.reals);
	buff.reals = NULL;
	_mm_free(buff.zImags);
	buff.zImags = NULL;
	_mm_free(buff.zReals);
	buff.zReals = NULL;
}

#pragma omp declare simd simdlen(64) aligned(zReals, zImags:64) uniform(imag) linear(zReals, zImags:1) inbranch
int line_mandelbrot(float real, float imag, float *zReals, float *zImags){
	float zReal = *(zReals);
	float zImag = *(zImags);

	float r2 = (zReal) * (zReal);
	float i2 = (zImag) * (zImag);

	*(zImags) =  2.0f * zReal * zImag + imag;
	*(zReals) = r2 - i2 + real;
	return r2 + i2 > 4.0f;
}

int * LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	const int lwidth = width;
	const int lheight = height;
	const int llimit = limit;

	// const int step = 1;

	int *pdata = buff.data;
	float *pImags = buff.imags;
	float *pReals = buff.reals;
	float *pzImags = buff.zImags;
	float *pzReals = buff.zReals;
	int *pMask = buff.mask;

	int i, k, j, col;
	int rowIndex;
	int *pRow;

	float imag;
	int maskSum;

	for (i = 0; i < lheight; i++)
	{
		imag = *(pImags);
		pzReals = buff.zReals;
		pzImags = buff.zImags;
		pReals = buff.reals;
		pMask = buff.mask;
		#pragma omp simd simdlen(64) aligned(pzImags:64, pzReals:64, pReals:64, pMask:64)\
		linear(pzImags:1,pzReals:1,pReals:1,pMask:1)
		for(col = 0; col < lwidth; col++){
			*(pzImags++) = imag;
			*(pzReals++) = *(pReals++);
			*(pMask++) = 0;
		}
		maskSum = 0;
		// pRow = (pdata + (lwidth * i));
		// for (k = 0; k < llimit; ++k)
		for (k = 0; k < llimit; ++k)
		{
			if (maskSum == width) break;

			pzReals = buff.zReals;
			pzImags = buff.zImags;
			pReals = buff.reals;
			pMask = buff.mask;
			pRow = (buff.data + (lwidth * i));

			#pragma omp simd aligned(pRow:64, pReals:64, pzReals:64, pzImags:64, pMask:64) reduction(+:maskSum) linear(pRow, pReals, pzReals, pzImags, pMask) simdlen(64)
			for (j = 0; j < lwidth; j++)
			{
				// int mask = *(pMask);
				// if (!*(pMask)) {
				// 	// float zReal = *(pzReals);
				// 	// float zImag = *(pzImags);
				// 	// float real = *(pReals);

				// 	// float r2 = (zReal) * (zReal);
				// 	// float i2 = (zImag) * (zImag);

				// 	// //check and set current value
				// 	// int reached = r2 + i2 > 4.0f;
				// 	// *(pRow) = reached ? k : llimit;
				// 	// maskSum = maskSum + reached ? 1 : 0;
				// 	// *(pMask) = reached ? 1 : 0;

				// 	// //prepare for next iteration of mandelbrot
				// 	// //zImag = 2.0f * zReal * zImag + imag;
				// 	// //zReal = r2 - i2 + real;
				// 	// *(pzImags) =  2.0f * zReal * zImag + imag;
				// 	// *(pzReals) = r2 - i2 + real;
					
				// 	int result = mandelbrot(*pReals, imag, pzReals, pzImags);
				// 	*(pdata) = *(pMask) ? k : limit;
				// }
				if (!(*pMask)){
					int update = line_mandelbrot(*pReals, imag, pzReals, pzImags);
					*(pRow) = update ? k : limit;
					maskSum += update;
					*(pMask) = update;
				}
				pzReals++;
				pzImags++;
				pReals++;
				pMask++;
				pRow++;
			}
		}
		pImags++;
	}
	return buff.data;
}
