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
	blockSizeL3 = 512;
	blockSizeL2 = 256;
	blockSizeL1 = 128;
	
	// @TODO allocate & prefill memory
	data = (int *)(_mm_malloc(height * width * sizeof(int), 64));
	mask = (int *)(_mm_malloc(blockSizeL2 * sizeof(int), 64));
	for(size_t i = 0; i < height * width; i++){
		*(data + i) = limit;
	}
	imags = (float *)(_mm_malloc(height * sizeof(float), 64));
	reals = (float *)(_mm_malloc(width * sizeof(float), 64));
	zImags = (float *)(_mm_malloc(blockSizeL2 * sizeof(float), 64));
	zReals = (float *)(_mm_malloc(blockSizeL2 * sizeof(float), 64));

	for(size_t i = 0; i < height; i++){
		*(imags + i) = y_start + i * dy;
	}
	for(size_t j = 0; j < width; j++){
		*(reals + j) = x_start + j * dx;
	}

	maskSum = 0;
}

BatchMandelCalculator::~BatchMandelCalculator() {
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

// #pragma omp declare simd simdlen(64) aligned(zReals, zImags:64) uniform(imag) linear(zReals, zImags:1) inbranch
// int batch_mandelbrot(float real, float imag, float *zReals, float *zImags){
// 	float zReal = *(zReals);
// 	float zImag = *(zImags);

// 	float r2 = (zReal) * (zReal);
// 	float i2 = (zImag) * (zImag);

// 	*(zImags) =  2.0f * zReal * zImag + imag;
// 	*(zReals) = r2 - i2 + real;
// 	return r2 + i2 > 4.0f;
// }


int * BatchMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers

	int *pdata = data;
	float *pImags = imags;
	float *pReals = reals;
	float *pzImags = zImags;
	float *pzReals = zReals;
	int *pMask = mask;

	size_t i;
	size_t j;
	size_t k;
	size_t index;
	int rowIndex;
	int *pRow;

	float imag;
	float real;
	
	for (size_t blockHL3 = 0; blockHL3 < height/blockSizeL3; blockHL3++)
	{
		for(size_t blockWL3 = 0; blockWL3 < width/blockSizeL3; blockWL3++)
		{

			for (size_t blockHL2 = 0; blockHL2 < blockSizeL3/blockSizeL2; blockHL2++)
			{
				for (size_t blockWL2 = 0; blockWL2 < blockSizeL3/blockSizeL2; blockWL2++)
				{

					const size_t block_x = blockWL3 * blockSizeL3 + blockWL2 * blockSizeL2;
					const size_t block_y = blockHL3 * blockSizeL3 + blockHL2 * blockSizeL2;
					for (i = 0; i < blockSizeL2; i++)
						{
							int iG = block_y + i;
							imag = *(pImags + iG);
							pzReals = zReals;
							pzImags = zImags;
							pReals = (reals + block_x);
							pMask = mask;
							#pragma omp simd simdlen(64) aligned(pzImags:64, pzReals:64, pReals:64, pMask:64)\
							linear(pzImags:1,pzReals:1,pReals:1,pMask:1)
							for(index = 0; index < blockSizeL2; index++){
								*(pzImags++) = imag;
								*(pzReals++) = *(pReals++);
								*(pMask++) = 0;
							}
							maskSum = 0;
							// pRow = (pdata + (lwidth * i));
							// for (k = 0; k < llimit; ++k)
							for (k = 0; k < limit; ++k)
							{
								if (maskSum == blockSizeL2) break;

								pzReals = zReals;
								pzImags = zImags;
								pReals = (reals + block_x);
								pMask = mask;
								// pRow = (data + (lwidth * i));

								#pragma omp simd aligned(pReals:64, pzReals:64, pzImags:64, pMask:64) reduction(+:maskSum) linear(pReals, pzReals, pzImags, pMask) simdlen(64)
								for (j = 0; j < blockSizeL2; j++)
								{
									int jG = block_x + j;
									// if (!(*pMask)){
									// 	int update = batch_mandelbrot(*pReals, imag, pzReals, pzImags);
									// 	*(data + (iG * width + jG)) = update ? k : limit;
									// 	maskSum += update;
									// 	*(pMask) = update;
									// }
									if (!(*pMask)){
										float zReal = *(pzReals);
										float zImag = *(pzImags);

										float r2 = (zReal) * (zReal);
										float i2 = (zImag) * (zImag);

										bool reached = r2 + i2 > 4.0f;

										*(data + (iG * width + jG)) = reached ? k : limit;
										maskSum += reached;
										*(pMask) = reached;

										*(pzImags) =  2.0f * zReal * zImag + imag;
										*(pzReals) = r2 - i2 + *(pReals);
									}
									pzReals++;
									pzImags++;
									pReals++;
									pMask++;
								}
							}
						}
				}
			}
		}
	}
	
	return data;
}