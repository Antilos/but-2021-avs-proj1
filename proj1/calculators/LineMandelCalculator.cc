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
		data[i] = limit;
	}
	imags = (float *)(_mm_malloc(height * sizeof(float), 64));
	reals = (float *)(_mm_malloc(width * sizeof(float), 64));
	zImags = (float *)(_mm_malloc(width * sizeof(float), 64));
	zReals = (float *)(_mm_malloc(width * sizeof(float), 64));

	// for(size_t i = 0, j = 0; i < height || j < width; i++, j++){
	// 	*(mask + j) = 0;
	// 	*(imags + i) = y_start + i * dy;
	// 	*(reals + j) = x_start + j * dx;
	// }

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
	int *pdata = data;
	float *pImags = imags;
	float *pReals = reals;
	float *pzImags = zImags;
	float *pzReals = zReals;
	int *pMask = mask;
	for (size_t i = 0; i < height; i++)
	{
		for(size_t col = 0; col < width; col++){
			*(pzImags + col) = *(pImags + i);
			*(pzReals + col) = *(pReals + col);
			*(pMask + col) = 0;
		}
		maskSum = 0;
		for (size_t k = 0; maskSum < width && k < limit; ++k)
		{
			#pragma omp simd simdlen(64) aligned(pdata, pReals, pImags, pzImags, pzReals, pMask:64) reduction(+:maskSum)
			for (size_t j = 0; j < width; j++)
			{
				int dataIndex = (height * j + i);

				float zReal = *(pzReals + j);
				float zImag = *(pzImags + j);
				float real = *(pReals + j);
				float imag = *(pImags + i);

				float r2 = (zReal) * (zReal);
				float i2 = (zImag) * (zImag);

				//check and set current value
				int reached = r2 + i2 > 4.0f;
				int mask = *(pMask + j);
				int update = !(mask) && reached;
				*(pdata + dataIndex) = update ? k : *(pdata + dataIndex); //set value to current iteration
				maskSum += update ? 1 : 0;
				*(pMask + j) = (mask || update) ? 1 : 0;

				//prepare for next iteration of mandelbrot
				//zImag = 2.0f * zReal * zImag + imag;
				//zReal = r2 - i2 + real;
				*(pzImags + j) =  2.0f * zReal * zImag - imag;
				*(pzReals + j) = r2 - i2 + real;

			}
		}
	}
	return data;
}
