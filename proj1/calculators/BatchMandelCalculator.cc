/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	mandelMask = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	for(size_t i = 0; i < height * width; i++){
		data[i] = limit;
		mandelMask[i] = 0;
	}
	zImags = (float *)(_mm_malloc(height * width * sizeof(float), 64));
	zReals = (float *)(_mm_malloc(height * width * sizeof(float), 64));
}

BatchMandelCalculator::~BatchMandelCalculator() {
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


int * BatchMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	constexpr size_t blockSize = 64;
	int *pdata = data;
	for (size_t blockH = 0; blockH < height/blockSize; blockH++)
	{
		for(size_t blockW = 0; blockW < width/blockSize; blockW++)
		{
			for (size_t i = 0; i < blockSize; i++)
			{
				const size_t iGlobal = blockH * blockSize + i;
				//size_t maskSum = 0;
				//for (size_t k = 0; maskSum >= (int)width/blocksize || k < blockSize; ++k)
				for (size_t k = 0; k < limit; ++k)
				{
					#pragma omp simd simdlen(64)
					for (size_t j = 0; j < blockSize; j++)
					{
						const size_t jGlobal = blockW * blockSize + j;
						//Current position
						float y = y_start + iGlobal * dy; // current imaginary value
						float x = x_start + jGlobal * dx; // current real value

						//Current value
						zImags[width * iGlobal + jGlobal] = (k == 0 ? y : zImags[width * iGlobal + jGlobal]);
						zReals[width * iGlobal + jGlobal] = (k == 0 ? x : zReals[width * iGlobal + jGlobal]);

						float r2 = zReals[width * iGlobal + jGlobal] * zReals[width * iGlobal + jGlobal];
						float i2 = zImags[width * iGlobal + jGlobal] * zImags[width * iGlobal + jGlobal];

						//check and set current value
						//pdata[width * i + j] = (k + 1 == limit && !mandelMask[width * i + j]) ? limit : ((!mandelMask[width * i + j] && (r2 + i2 > 4.0f)) ? k : pdata[width * i + j]); //check wether we've reached limit
						pdata[width * iGlobal + jGlobal] = (!mandelMask[width * iGlobal + jGlobal] && (r2 + i2 > 4.0f)) ? k : pdata[width * iGlobal + jGlobal]; //set value to current iteration
						mandelMask[width * iGlobal + jGlobal] = (mandelMask[width * iGlobal + jGlobal] || (r2 + i2 > 4.0f)) ? 1 : 0;
						//maskSum += mandelMask[width * i + j] ? 1 : 0;

						//prepare for next iteration of mandelbrot
						zImags[width * iGlobal + jGlobal] =  2.0f * zReals[width * iGlobal + jGlobal] * zImags[width * iGlobal + jGlobal] + y;
						zReals[width * iGlobal + jGlobal] = r2 - i2 + x;

					}
				}
			}
		}
	}
	
	return data;
}