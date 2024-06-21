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
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
}


int * LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int *pdata = data;
	float x = x_start;
	float y = y_start;
	float zReal = x;
	float zImag = y;
	for (int i = 0; i < height; i++)
	{
		y += i * dy;
		for (int k = 0; k < limit; ++k)
		{
			zImag = y;
			for (int j = 0; j < width; j++)
			{
				x += j * dx; // current real value
				zReal = x;
				//float y = y_start + i * dy; // current imaginary value
				r2 = zReal * zReal;
				i2 = zImag * zImag;
				pdata[i*width + j] = r2 + i2 > 4.0f ? k : limit
			}
			zImag = 2.0f * zReal * zImag + y;
			zReal = zReal * zReal
		}
	}
	return data;
	return NULL;
}
