/**
 * @file LineMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */

#ifndef LINEMANDELCALCULATOR_H
#define LINEMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    // @TODO add all internal parameters
    int *data;
    int *mask; //did we reach target in this column?
    float *imags;
    float *reals;
    float *zImags; //current zImag
    float *zReals; //current zReal

    size_t maskSum = 0;
};
#endif