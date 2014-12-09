/*
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a preconditioned conjugate gradient solver on 
 * the GPU using CUBLAS and CUSPARSE.  Relative to the conjugateGradient
 * SDK example, this demonstrates the use of cusparseScsrsv_solve()
 * for solving triangular systems.  Specifically, the preconditioned 
 * conjugate gradient method with an Incomplete Cholesky preconditioner
 * is used to solve a randomly generated symmetric, strictly diagonally 
 * dominant, tridiagonal real matrix.
 *
 * Note that the code in this example and the specific matrices used here
 * were chosen to demonstrate the use of the CUSPARSE library as simply
 * and as clearly as possible.  This is not optimized code and the input 
 * matrices have been artificially generated, hence this code should not
 * be used either as a performance guide or for benchmarking purposes. 
 * In particular, the efficient solution of tridiagonal systems is a well
 * researched topic and interested users are encouraged to consult the
 * following references:
 *
 *    Efficient Tridiagonal Solvers for ADI methods and Fluid Simulation:
 *    http://nvidia.fullviewmedia.com/gtc2010/0921-a7-2015.html
 * 
 *    Tridiagonal Solvers: Auto-Tuning and Optimizations
 *    http://nvidia.fullviewmedia.com/gtc2010/0921-c-2085.html
 *   
 */


// Utilities and system includes
#include <shrUtils.h>
#include <shrQATest.h>
#include <cutil_inline.h>
#include <cusparse.h>
#include <cublas.h>

const char * sSDKname	  = "conjugateGradientPrecond";


/* checkStatus: concise method for verifying CUDA return status */
int checkStatus ( cusparseStatus_t status, char *msg )
{
    if ( status != CUSPARSE_STATUS_SUCCESS ) {
        fprintf (stderr, "!!!! CUSPARSE %s ERROR \n", msg);
        return 1;
    }
    return 0;
}


/* genICP: Generate the Incomplete Cholesky Preconditioner for a symmetric tridiagonal.  
   Follows description from Golub & Van Loan, "Matrix Computations 3rd Ed.", section 10.3.2 */
void genICP ( int *rowPtrs, float *vals, int N, int *colIndsICP, int *rowPtrsICP, float *valsICP )
{
    // Define a lower triangular banded matrix with 2 bands.
    rowPtrsICP[0] = 0;
    colIndsICP[0] = 0;
    int inz = 1;
    for ( int k=1; k<N; k++ ) {
        rowPtrsICP[k] = inz;
        for ( int j=k-1; j<=k; j++ ) {
            colIndsICP[inz] = j;
            inz++;
        }
    }
    rowPtrsICP[N] = inz;

    // copy A into H
    valsICP[0] = vals[0];
    for ( int k=1; k<N; k++ ) {
        valsICP[rowPtrsICP[k]] = vals[rowPtrs[k]];
        valsICP[rowPtrsICP[k]+1] = vals[rowPtrs[k]+1];    
    }

    // construct H
    for ( int k=0; k<N; k++ ) {
        valsICP[rowPtrsICP[k+1]-1] = sqrt(valsICP[rowPtrsICP[k+1]-1]);
        if ( k < N-1 ) {
            valsICP[rowPtrsICP[k+1]] /= valsICP[rowPtrsICP[k+1]-1];      
            valsICP[rowPtrsICP[k+1]+1] -= valsICP[rowPtrsICP[k+1]]*valsICP[rowPtrsICP[k+1]];
        }
    }

    return;
}


/* genTridiag: Generate a random, tridiagonal, symmetric, strictly diagonally dominant matrix. */
void genTridiag(int *I, int *J, float *val, int M, int N, int nz)
{
    assert (M==N);
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;
    for (int i = 1; i < N; i++) {
        if (i > 1) 
            I[i] = I[i-1]+3;
        else 
            I[1] = 2;
        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;
        if (i < N-1) 
            J[start+2] = i + 1;
        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;
        if (i < N-1) 
            val[start+2] = (float)rand()/RAND_MAX;
    }
    I[N] = nz;
}

/* Solve Ax=b using the conjugate gradient method a) without any preconditioning and b) using an Incomplete Cholesky preconditioner. */
int main(int argc, char **argv)
{
    const int max_iter = 100;
    int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    int *d_colIndsICP, *d_rowPtrsICP;
    int qatest = 0;
    const float tol = 1e-8f;                          
    float *x, *y, *rhs; 
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_zm1, *d_zm2, *d_rm2;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float *d_valsICP;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;

	int nErrors = 0;

    shrQAStart(argc, argv);

    shrSetLogFileName ("preconditionedConjugateGradient.txt");

    // QA testing mode
    if ( cutCheckCmdLineFlag (argc, (const char**)argv, "qatest")) {
        qatest = 1;
    }

    // This will pick the best possible CUDA capable device
    cudaDeviceProp deviceProp;
    int devID = cutilChooseCudaDevice(argc, argv);
    if (devID < 0) {
        printf("exiting...\n");
        cutilExit(argc, argv);
        exit(EXIT_SUCCESS);
    }
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, devID) );

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n", 
	   deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);
    if(version < 0x11) 
    {
       printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
       cutilDeviceReset();
       shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
    }

    /* Generate a random tridiagonal symmetric matrix in CSR (Compressed Sparse Row) format */
    M = N = 4096;
    nz = (N-2)*3 + 4;
    I = (int*)malloc(sizeof(int)*(N+1));                               // csr row pointers for matrix A
    J = (int*)malloc(sizeof(int)*nz);                                  // csr column indices for matrix A
    val = (float*)malloc(sizeof(float)*nz);                            // csr values for matrix A
    genTridiag(I, J, val, M, N, nz);

    x = (float*)malloc(sizeof(float)*N);
    y = (float*)malloc(sizeof(float)*N);
    rhs = (float*)malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++) {
	rhs[i] = 1.0;                                                  // Initialize RHS to 1.0
	x[i] = 0.0;                                                    // Initial approximation of solution
    }

    cusparseHandle_t handle = 0;
    cusparseStatus_t status;

    /* Get handle to the CUSPARSE context */
    status = cusparseCreate(&handle);
    if ( checkStatus ( status, (char*)"initialization" ) ) return EXIT_FAILURE;

    /* Description of the A matrix*/
    cusparseMatDescr_t descr = 0;
    status = cusparseCreateMatDescr(&descr); 
    if ( checkStatus ( status, (char*)"cusparseCreateMatDescr" ) ) return EXIT_FAILURE;
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    cutilSafeCall( cudaMalloc((void**)&d_col, nz*sizeof(int)) );
    cutilSafeCall( cudaMalloc((void**)&d_row, (N+1)*sizeof(int)) );
    cutilSafeCall( cudaMalloc((void**)&d_val, nz*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void**)&d_x, N*sizeof(float)) );  
    cutilSafeCall( cudaMalloc((void**)&d_y, N*sizeof(float)) );  
    cutilSafeCall( cudaMalloc((void**)&d_r, N*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void**)&d_p, N*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void**)&d_omega, N*sizeof(float)) );

    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);

    /* Preconditioned Conjugate Gradient.  
       Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */

    shrLog("Convergence of conjugate gradient without preconditioning: \n");
    k = 0;
    r0 = 0;
    r1 = cublasSdot(N, d_r, 1, d_r, 1);
    while (r1 > tol*tol && k <= max_iter) {
        k++;
        if ( k == 1 ) {
            cublasScopy (N, d_r, 1, d_p, 1);
        }
        else {
            beta = r1/r0;
            cublasSscal(N, beta, d_p, 1);         
            cublasSaxpy (N, 1.0, d_r, 1, d_p, 1) ;
        }
        cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, 1.0, 
                        descr, d_val, d_row, d_col, d_p, 0.0, d_omega);
        alpha = r1/cublasSdot (N, d_p, 1, d_omega, 1);
        cublasSaxpy (N, alpha, d_p, 1, d_x, 1);
        cublasSaxpy (N, -alpha, d_omega, 1, d_r, 1);
        r0 = r1;
        r1 = cublasSdot(N, d_r, 1, d_r, 1);
        shrLog("  iteration = %3d, residual = %e \n", k, sqrt(r1));
    }

    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    err = 0.0;
    for (int i = 0; i < N; i++) {
        rsum = 0.0;
        for (int j = I[i]; j < I[i+1]; j++) {
            rsum += val[j]*x[J[j]];
        }
        diff = fabs(rsum - rhs[i]);
        if (diff > err) err = diff;
    }

    shrLog("  Convergence Test: %s, maximum single component error in Ax-b = %e\n", (k <= max_iter) ? "OK" : "FAIL", err);
	nErrors += (k > max_iter) ? 1 : 0;
	qaerr1 = err;

    shrLog("\nConvergence of conjugate gradient using Incomplete Cholesky preconditioning: \n");

    // create a matrix description for the preconditioner M
    cusparseMatDescr_t descrM = 0;
    status = cusparseCreateMatDescr(&descrM); 
    if ( checkStatus ( status, (char*)"cusparseCreateMatDescr" ) ) return EXIT_FAILURE;
    cusparseSetMatType ( descrM, CUSPARSE_MATRIX_TYPE_TRIANGULAR );
    cusparseSetMatIndexBase ( descrM, CUSPARSE_INDEX_BASE_ZERO );
    cusparseSetMatDiagType ( descrM, CUSPARSE_DIAG_TYPE_NON_UNIT );
    cusparseSetMatFillMode ( descrM, CUSPARSE_FILL_MODE_LOWER );

    // create the Incomplete Cholesky Preconditioner matrices H and H^T. 
    int *colIndsICP = NULL;
    int *rowPtrsICP = NULL;
    float *valsICP  = NULL;
    int nzICP = 2*N-1;
    colIndsICP = (int *) malloc (nzICP*sizeof(int));
    rowPtrsICP = (int *) malloc ((N+1)*sizeof(int));
    valsICP = (float *) malloc (nzICP*sizeof(float));

    // generate the Incomplete Cholesky factor H (lower triangular) for the matrix A.
    genICP ( I, val, N, colIndsICP, rowPtrsICP, valsICP );

    cutilSafeCall( cudaMalloc((void**)&d_colIndsICP, nzICP*sizeof(int)) );
    cutilSafeCall( cudaMalloc((void**)&d_valsICP, nzICP*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void**)&d_rowPtrsICP, (N+1)*sizeof(int)) );
    cutilSafeCall( cudaMalloc((void**)&d_zm1, (N)*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void**)&d_zm2, (N)*sizeof(float)) );
    cutilSafeCall( cudaMalloc((void**)&d_rm2, (N)*sizeof(float)) );
    cudaMemcpy(d_colIndsICP, colIndsICP, nzICP*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtrsICP, rowPtrsICP, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valsICP, valsICP, nzICP*sizeof(float), cudaMemcpyHostToDevice);

    // create the analysis info object for the Non-Transpose case
    cusparseSolveAnalysisInfo_t info = 0;
    status = cusparseCreateSolveAnalysisInfo(&info);
    if ( checkStatus ( status, (char*)"cusparseCreateSolveAnalysisInfo" ) ) return EXIT_FAILURE;

    // create the analysis info object for the Transpose case
    cusparseSolveAnalysisInfo_t infoTrans = 0;
    status = cusparseCreateSolveAnalysisInfo(&infoTrans);
    if ( checkStatus ( status, (char*)"cusparseCreateSolveAnalysisInfo Trans" ) ) return EXIT_FAILURE;

    // Perform the analysis for the Non-Transpose case
    status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, descrM, d_valsICP, d_rowPtrsICP, d_colIndsICP, info);
    if ( checkStatus ( status, (char*)"susparseScsrv_analysis" ) ) return EXIT_FAILURE;

    // Perform the analysis for the Transpose case
    status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, N, descrM, d_valsICP, d_rowPtrsICP, d_colIndsICP, infoTrans);
    if ( checkStatus ( status, (char*)"susparseScsrv_analysis Trans" ) ) return EXIT_FAILURE;

    for (int i = 0; i < N; i++) {
	x[i] = 0.0; 
    }

    cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

    /* Preconditioned Conjugate Gradient.  
       Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Algorithm 10.3.1  */
    k = 0;
    r1 = cublasSdot(N, d_r, 1, d_r, 1);
    while (r1 > tol*tol && k <= max_iter) {
        // solve M z = H H^T z = r

        // Forward Solve: H y = r
        status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, 1.0, descrM, 
			              d_valsICP, d_rowPtrsICP, d_colIndsICP, info, d_r, d_y);    
        if ( checkStatus ( status, (char*)"susparseScsrv_solve" ) ) return EXIT_FAILURE;
        // Back Substitution: H^T z = y
        status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, N, 1.0, descrM, 
			              d_valsICP, d_rowPtrsICP, d_colIndsICP, infoTrans, d_y, d_zm1);    
        if ( checkStatus ( status, (char*)"susparseScsrv_solve" ) ) return EXIT_FAILURE;
          
        k++;

        if ( k == 1 ) {
            cublasScopy (N, d_zm1, 1, d_p, 1);
        }
        else {
            beta = cublasSdot(N, d_r, 1, d_zm1, 1)/cublasSdot(N, d_rm2, 1, d_zm2, 1);
            cublasSscal(N, beta, d_p, 1);    
            cublasSaxpy (N, 1.0, d_zm1, 1, d_p, 1) ;
        }
        cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, 1.0, descr, d_val, d_row, d_col, d_p, 0.0, d_omega);
        alpha = cublasSdot(N, d_r, 1, d_zm1, 1)/cublasSdot (N, d_p, 1, d_omega, 1);
        cublasSaxpy (N, alpha, d_p, 1, d_x, 1);
        cublasScopy (N, d_r, 1, d_rm2, 1);
        cublasScopy (N, d_zm1, 1, d_zm2, 1);
        cublasSaxpy (N, -alpha, d_omega, 1, d_r, 1);
        r1 = cublasSdot(N, d_r, 1, d_r, 1);
        shrLog("  iteration = %3d, residual = %e \n", k, sqrt(r1));
    }

    cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    // check result
    err = 0.0;
    for (int i = 0; i < N; i++) {
        rsum = 0.0;
        for (int j = I[i]; j < I[i+1]; j++) {
            rsum += val[j]*x[J[j]];
        }
        diff = fabs(rsum - rhs[i]);
        if (diff > err) err = diff;
    }

    shrLog("  Convergence Test: %s, maximum single component error in Ax-b = %e\n\n", (k <= max_iter) ? "OK" : "FAIL", err);
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr2 = err;

    cusparseDestroy(handle);

    free(I);
    free(J);
    free(val);
    free(x);
    free(y);
    free(rhs);
    free(colIndsICP);
    free(rowPtrsICP);
    free(valsICP);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_omega);
    cudaFree(d_colIndsICP);
    cudaFree(d_rowPtrsICP);
    cudaFree(d_valsICP);
    cudaFree(d_zm1);
    cudaFree(d_zm2);
    cudaFree(d_rm2);

    cutilDeviceReset();

    shrLog("  Test Summary:\n");
    shrLog("     Counted total of %d errors\n", nErrors);
    shrLog("     qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2) );
    shrQAFinishExit(argc, (const char **)argv, ((nErrors == 0) &&fabs(qaerr1)<1e-5 && fabs(qaerr2) < 1e-5) ? QA_PASSED : QA_FAILED);
}

