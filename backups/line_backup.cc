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
	mask = (int *)(_mm_malloc(width * sizeof(int), 64));
	for(size_t i = 0; i < height * width; i++){
		*(data + i) = limit;
	}
	imags = (float *)(_mm_malloc(height * sizeof(float), 64));
	reals = (float *)(_mm_malloc(width * sizeof(float), 64));
	zImags = (float *)(_mm_malloc(width * sizeof(float), 64));
	zReals = (float *)(_mm_malloc(width * sizeof(float), 64));

	for(size_t i = 0; i < height; i++){
		*(imags + i) = y_start + i * dy;
	}
	for(size_t j = 0; j < width; j++){
		*(reals + j) = x_start + j * dx;
		*(mask + j) = 0;
	}

	maskSum = 0;
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
	_mm_free(mask);
	mask = NULL;
	_mm_free(imags);
	imags = NULL;
	_mm_free(reals);
	reals = NULL;
	_mm_free(zImags);
	zImags = NULL;
	_mm_free(zReals);
	zReals = NULL;
}

int * LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	const int lwidth = width;
	const int lheight = height;
	const int llimit = limit;

	const int step = 1;

	int *pdata;// = data;
	float *pImags = imags;
	float *pReals = reals;
	float *pzImags = zImags;
	float *pzReals = zReals;
	int *pMask = mask;

	int i, k, j, col;
	int rowIndex;
	int *pRow;

	float imag;

	for (i = 0; i < lheight; i++)
	{
		imag = *(pImags);
		pzReals = zReals;
		pzImags = zImags;
		pReals = reals;
		pMask = mask;
		#pragma omp simd simdlen(64) aligned(pzImags:64, pzReals:64, pReals:64, pMask:64)\
		linear(pzImags:1,pzReals:1,pReals:1,pMask:1)
		for(col = 0; col < lwidth; col++){
			*(pzImags++) = imag;
			*(pzReals++) = *(pReals++);
			*(pMask++) = 0;
		}
		maskSum = 0;
		// pRow = (pdata + (lwidth * i));
		for (k = 0; maskSum < lwidth && k < llimit; ++k)
		{
			pzReals = zReals;
			pzImags = zImags;
			pReals = reals;
			pMask = mask;
			pRow = (data + (lwidth * i));
			#pragma omp simd simdlen(64)\
			aligned(pdata:64, pReals:64, pImags:64, pzImags:64, pzReals:64, pMask:64, pRow:64)\
			linear(pzReals:step, pzImags:step, pReals:step, pMask:step, pRow:step)\
			reduction(+:maskSum)
			for (j = 0; j < lwidth; j++)
			{
				int mask = *(pMask);
				if (!mask) {
					float zReal = *(pzReals);
					float zImag = *(pzImags);
					float real = *(pReals);

					float r2 = (zReal) * (zReal);
					float i2 = (zImag) * (zImag);

					//check and set current value
					int reached = r2 + i2 > 4.0f;
					*(pRow) = reached ? k : llimit;
					maskSum += reached ? 1 : 0;
					*(pMask) = reached ? 1 : 0;

					//prepare for next iteration of mandelbrot
					//zImag = 2.0f * zReal * zImag + imag;
					//zReal = r2 - i2 + real;
					*(pzImags) =  2.0f * zReal * zImag + imag;
					*(pzReals) = r2 - i2 + real;
				}
				pzReals += step;
				pzImags += step;
				pReals += step;
				pMask += step;
				pRow += step;
			}
		}
		pImags += step;
	}
	return data;
}
