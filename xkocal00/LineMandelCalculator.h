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
    typedef struct buffer_struct{
        int maskSum;
        float *zImags;
        float *zReals;
        float *imags;
        float *reals;
        int *mask;
        int *data;
    } t_helper_buffer;
    t_helper_buffer buff;
    // typedef struct posstruct{
    //     float *zImags;
    //     float *zReals;
    // } t_posstruct;
    // // @TODO add all internal parameters
    // int *data;
    // int *mask; //did we reach target in this column?

    // // float *imags;
    // // float *reals;
    // t_posstruct pos;

    // // float *zImags; //current zImag
    // // float *zReals; //current zReal
    // t_zstruct z;

    size_t maskSum = 0;
};
#endif