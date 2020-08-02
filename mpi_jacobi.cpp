/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */

void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
/* INFO of the communication topology */
	int rank, p, rank00;
	MPI_Comm_size(comm, &p); /* get p */
	MPI_Comm_rank(comm, &rank); /* get rank */
	int q = int (sqrt(p)); /* get q, where p=q*q */
	int coordinate[2]; 
	MPI_Cart_coords(comm, rank, 2, &coordinate[0]); /* get coords of proc. grids */
	int coord00[2] = {0, 0};
	MPI_Cart_rank(comm, coord00, &rank00); /* get rank of proc. (0,0) */
	
	/* Create a new comm for first column */
	int color = 1;
	if (coordinate[1]==0) color = 0;
	MPI_Comm comm_column;
	MPI_Comm_split(comm, color, 0, &comm_column);
	
	/* Scatter from process (0,0) to * processes (i,0) */
	if (color==0) {
		int sendcounts[q], displs[q], recvcounts;
		for (int i=0; i<q; i++) {
			sendcounts[i] = block_decompose(n, q, i); /* get sendcounts */
			displs[i] = i==0? 0:displs[i-1]+sendcounts[i-1]; /* get displacements */
		}		
		recvcounts = block_decompose(n, q, coordinate[0]); /* get recvcounts */
		*local_vector = new double[recvcounts]; /* prepare space for recv */

		MPI_Scatterv(&input_vector[0], sendcounts, displs, MPI_DOUBLE, *local_vector, recvcounts, MPI_DOUBLE, rank00, comm_column);
	}
	MPI_Comm_free(&comm_column); /* free sub communicator */
}



void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
	
	/*get dimension of mesh grid, p=q*q */
	int grid_p;
	MPI_Comm_size(comm,&grid_p);
	int grid_q=(int) sqrt(grid_p);

	/*get rank and coordinate*/
	int localRank;
	MPI_Comm_rank(comm,&localRank);
	int Coords[2];
	MPI_Cart_coords(comm,localRank,2,Coords);
	
	/*get local vector size*/
	int local_len=block_decompose(n, grid_q, Coords[0]);
	
	/*build sub_community based on the column*/
	MPI_Comm sub_comm;
    	MPI_Comm_split(comm, Coords[1], Coords[0], &sub_comm);
    	int sub_rank; 
	MPI_Comm_rank(sub_comm, &sub_rank);

	/*send and recieve vectors from (i,0) to (0,0)*/
	if (Coords[1]==0){
		int* recvCounts = new int[grid_q];
		int* recvDisp = new int[grid_q];
		recvDisp[0]=0;
		for (int i=0;i<grid_q;i++){
			recvCounts[i] = block_decompose(n, grid_q, i);
            		if (i>0) recvDisp[i] = recvDisp[i - 1] + recvCounts[i - 1];
		}
		MPI_Gatherv(local_vector, local_len, MPI_DOUBLE, output_vector, recvCounts, recvDisp, MPI_DOUBLE, 0, sub_comm);
		delete recvCounts;
    		delete recvDisp;
	}
    	MPI_Comm_free(&sub_comm);
}


void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
/* INFO of the communication topology */
	int rank, p, rank00;
	MPI_Comm_size(comm, &p); /* get p */
	MPI_Comm_rank(comm, &rank); /* get rank */
	int q = int (sqrt(p)); /* get q, where p=q*q */
	int coordinate[2]; 
	MPI_Cart_coords(comm, rank, 2, &coordinate[0]); /* get coords of proc. grids */
	
	int coord00[2] = {0, 0};
	MPI_Cart_rank(comm, coord00, &rank00); /* get rank of proc. (0,0) */
	
	int m = block_decompose(n, q, 0); /* get ceil(n/q) */
	int l = q*m; /* q*ceil(n/q) which is the guaranteed large enough size */
	double* A=new double[l*l]; /* create matrix A for storing connected blocks to be sent */
	int rank_curr, coord_curr[2], length_row, length_col, sendcounts[p], displs[p], displs_curr;
	int displs_input = 0;
	/* get info for scatter */
	for (int i=0; i<q; i++) {
		coord_curr[0] = i;
		length_row = block_decompose(n, q, i);
		for (int j=0; j<q; j++) {
			coord_curr[1] = j;
			MPI_Cart_rank(comm, coord_curr, &rank_curr); /* get rank of proc. (i,j) */		
			length_col = block_decompose(n, q, j);
			sendcounts[rank_curr] = length_row*length_col;
			/* we have input_matrix at processor (0,0) */
			if (rank==rank00) { 
				displs[rank_curr] = rank_curr*m*m;
				displs_curr = rank_curr*m*m;
				/*printf("\n rank=%d, rank_recv=%d,%d\n\n",rank,rank_curr,sendcounts[rank_curr]);*/
				/* get displs */
				for (int ii=0; ii<length_row; ii++) {
					displs_input = ii==0 ? displs_input : displs_input+n-length_col;
					for (int jj=0; jj< length_col; jj++) A[displs_curr++] = input_matrix[displs_input++];
				}
				displs_input = j==q-1 ? displs_input : displs_input-n*(length_row-1);
			}

		}
	}

	int recvcounts = block_decompose(n, q, coordinate[0])*block_decompose(n, q, coordinate[1]); /* get recvcounts */
	*local_matrix = new double[recvcounts];
	MPI_Scatterv(&A[0], sendcounts, displs, MPI_DOUBLE, *local_matrix, recvcounts, MPI_DOUBLE, rank00, comm); /* distribute matrix */
	/*printf("\n rank=%d, x0=%f\n\n",rank,local_matrix[0]);*/
	delete[] A; /* free the space created for A */
}

void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
	/*get dimension of mesh grid, p=q*q */
	int grid_p;
	MPI_Comm_size(comm,&grid_p);
	int grid_q=(int) sqrt(grid_p);

	/*get rank and coordinate*/
	int localRank;
	MPI_Comm_rank(comm,&localRank);
	int Coords[2];
	MPI_Cart_coords(comm,localRank,2,Coords);  
	int i=Coords[0], j=Coords[1];

	/*get local length of col*/
	int row_len = block_decompose(n, grid_q, i);
	int col_len = block_decompose(n, grid_q, j);
	
	/*send col_vector to diagonal processors*/
	if(i==0 && j==0) {
		for (int i =0 ; i<row_len; i++) row_vector[i]=col_vector[i];
	}
	if (j==0 && i!=0){
		int diag_cor [2]= {i,i};
		int diag_rank;
		MPI_Cart_rank(comm, diag_cor, &diag_rank);
		MPI_Send(&col_vector[0], row_len, MPI_DOUBLE, diag_rank,1,comm);
	}
	
	/*recieve the vector on diagnal processor*/
	if (i==j && i!=0){
		int left_cor [2]= {i,0};
		int left_rank;
		MPI_Cart_rank(comm,left_cor,&left_rank);
		MPI_Recv(&row_vector[0], row_len, MPI_DOUBLE, left_rank, 1, comm, MPI_STATUS_IGNORE);
	}

	/*split community into column community*/
	MPI_Comm sub_comm;
	MPI_Comm_split(comm, j, i, &sub_comm);

	/*broadcast from i,i to the column*/
	MPI_Bcast(row_vector, col_len, MPI_DOUBLE, j, sub_comm);
	MPI_Comm_free(&sub_comm);

}

void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
/* TODO */
/* INFO of the communication topology */
	int rank;
	MPI_Comm_rank(comm, &rank); /* get rank */
	int coordinate[2]; 
	MPI_Cart_coords(comm, rank, 2, &coordinate[0]); /* get coords of proc. grids */
	
	int length_row = block_decompose_by_dim(n, comm, 1);
	int length_col = block_decompose_by_dim(n, comm, 0);

	double row_vector[length_row], y[length_col];
	transpose_bcast_vector(n, local_x, row_vector, comm); /* transpose the vector */
	/* conduct the multiplication locally */
	for (int i=0; i<length_col; i++) {
		y[i] = 0;
		for (int j=0; j<length_row; j++) y[i] += local_A[length_row*i+j] * row_vector[j];
	}
			
	/* Create new comm for each row */
	MPI_Comm comm_row;
	MPI_Comm_split(comm, coordinate[0], coordinate[1], &comm_row);
	/* get the sum using Reduce at column processors using separate row communicator */
	MPI_Reduce(&y[0], &local_y[0], length_col, MPI_DOUBLE, MPI_SUM, 0, comm_row);

	MPI_Comm_free(&comm_row); /* free the sub communicator */

}


void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x, MPI_Comm comm, int max_iter, double l2_termination)
{
/* INFO of the communication topology */
	int rank, rank00, ranki0, rankii;
	MPI_Comm_rank(comm, &rank); /* get rank */
	int coordinate[2]; 
	MPI_Cart_coords(comm, rank, 2, &coordinate[0]); /* get coords of proc. grids */
	
	int coord00[2] = {0, 0}; 
	MPI_Cart_rank(comm, coord00, &rank00); /* get rank of proc. (0,0) */
	int coordi0[2] = {coordinate[0], 0}; 
	MPI_Cart_rank(comm, coordi0, &ranki0); /* get rank of proc. (i,0) */
	int coordii[2] = {coordinate[0], coordinate[0]};
	MPI_Cart_rank(comm, coordii, &rankii); /* get rank of proc. (i,i) */

	int length_row = block_decompose_by_dim(n, comm, 1);
	int length_col = block_decompose_by_dim(n, comm, 0);

	double D[length_col], R[length_row * length_col], local_y[length_col];
	/* get local D and local R */
	for (int i=0; i< length_col; i++) {
		for (int j=0; j< length_row; j++) R[length_row*i+j] = (i==j&&coordinate[0]==coordinate[1]) ? 0:local_A[length_row*i+j];
		local_x[i] = 0;
		if (coordinate[0]==coordinate[1]) D[i] = local_A[length_row*i+i];
	}
	/* send local D from processor (i,i) to (i,0) */
	if (coordinate[0]==coordinate[1]&&coordinate[0]!=0) MPI_Send(D, length_col, MPI_DOUBLE, ranki0, 20, comm);
    	if (coordinate [1]==0&&coordinate[0]!=0) MPI_Recv(D, length_col, MPI_DOUBLE, rankii, 20, comm, MPI_STATUS_IGNORE);
	
	double err = 2 * l2_termination; /* set a larger value to enter the loop */
	double local_err[length_col], global_err[n];
	int iter = 0; /* initialize the iteration counts */
	/* while loop for iterative computation */
	while (iter < max_iter && err > l2_termination) {
	
		distributed_matrix_vector_mult(n, R, local_x, local_y, comm); /* get local y=Rx */
		if (coordinate [1] == 0) { /* update local y=(b-Rx)/D on proc (i,0) */
			for (int i=0; i<length_col; i++) local_x[i] = (local_b[i]- local_y[i])/D[i];
		}

		distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm); /* get local y=Ax */
		if (coordinate [1] == 0) { /* get local error=(b-y)^2 on proc (i,0) */
			for (int i=0; i<length_col; i++) local_err[i] = (local_b[i]- local_y[i])*(local_b[i]- local_y[i]);
		}
		gather_vector(n, local_err, global_err, comm); /* get all error at proc (0,0) */
		if (rank == rank00) { /* at proc (0,0), we calculate the total error */
			err = 0;
			for (int i=0; i<n; i++) err += global_err[i];
			err = sqrt(err);	/* l2 error */
		}
		MPI_Bcast(&err, 1, MPI_DOUBLE, rank00, comm); /* share l2 error with all proc*/
		iter ++; /* iteration count increment */
	}	
}

void mpi_matrix_vector_mult(const int n, double* A, double* x, double* y, MPI_Comm comm)
{
/* distribute the array onto local processors! */
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);
/* allocate local result space */
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);
/* gather results back to rank 0 */
    gather_vector(n, local_y, y, comm);
}

void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm, int max_iter, double l2_termination)
{
/* distribute the array onto local processors! */
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);
/* allocate local result space */
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);
/* gather results back to rank 0 */
    gather_vector(n, local_x, x, comm);
}
