/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

/* my implementation:*/
#include <iostream>
#include <math.h>

/* Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x and y */
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
	for(int i=0; i<n; i++) {
		y[i] = 0; /* zero before summation */
		for(int j=0; j<n; j++) y[i] += A[i*n+j]*x[j]; /* add each product up */
	}
}

/* Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x and a n-dimensional vector y*/
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
	for(int i=0; i<n; i++) {
		y[i] = 0; /* zero before summation */
		for(int j=0; j<m; j++) y[i] += A[i*m+j]*x[j]; /* add each product up */
	}
}

/* implements the sequential jacobi method */
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
	/*declare matrix D=inv(diag(A)), soln vector y and matrix R=A-inv(D)*/
	double D[n], y[n], R[n*n]; 
	/*Copy A to R*/
	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) R[i*n+j]=A[i*n+j];
	}
	/*Initialize x, D and R*/
	for(int i=0; i<n; i++) {
		x[i] = 0;
		D[i] = 1/A[i*n+i];
		R[i*n+i] = 0;
	}
	/*Initialize l2 error as err*/
	double err = 2*l2_termination;
	/*Initialize iteration counts as iter*/
	int iter = 0;
	/*while err>l2_termination && iter<max_iter, do the following*/
	while (err>l2_termination && iter<max_iter) {
		iter += 1; /*iteration counts ++*/
		/*call MVM for R*x */
		matrix_vector_mult(n, &R[0], &x[0], &y[0]);
		/*compute 1/d*(b-R*x) */
		for(int i=0; i<n; i++) x[i] = D[i]*(b[i]-y[i]);
		/* call MVM for A*x to check current difference*/
		matrix_vector_mult(n, &A[0], &x[0], &y[0]);
		/* updated l2 error*/
		err = 0;
		for(int i=0; i<n; i++) err += pow(b[i]-y[i], 2);
		err = sqrt(err);
	}
}

