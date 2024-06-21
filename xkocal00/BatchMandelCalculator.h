/**
 * @file BatchMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

typedef struct zstruct{
    float *zImags;
    float *zReals;
} t_zstruct;

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    // @TODO add all internal parameters
    size_t blockSizeL3;
	size_t blockSizeL2;
    size_t blockSizeL1;

    int *data;
    int *mask; //did we reach target in this column?

    float *imags;
    float *reals;

    float *zImags; //current zImag
    float *zReals; //current zReal
    // t_zstruct z;

    size_t maskSum = 0;
};

#endif