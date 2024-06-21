/**
 * @file BatchMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    // @TODO add all internal parameters
    int *data;
    int *mandelValue; //side buffer for storing mandelbrot iterations while iterating over columns
    int *mandelMask; //did we reach target in this column?
    float *zImags; //current zImag
    float *zReals; //current zReal
};

#endif