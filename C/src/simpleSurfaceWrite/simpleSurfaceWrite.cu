/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
/*
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates 
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

// includes, kernels
#include <simpleSurfaceWrite_kernel.cu>

char *image_filename = "lena_bw.pgm";
char *ref_filename   = "ref_rotated.pgm";
float angle = 0.5f;    // angle to rotate image by (in radians)

static char *sSDKname = "simpleSurfaceWrite";

#define MIN_EPSILON_ERROR 5e-3f


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    bool bTestResult = true;
    shrQAStart(argc, argv);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
	int devID = cutilChooseCudaDevice(argc, argv);

    // get number of SMs on this GPU
    cudaDeviceProp deviceProps;

	cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors, SM %d.%d\n", deviceProps.name, deviceProps.multiProcessorCount, deviceProps.major, deviceProps.minor );

	if (deviceProps.major < 2) {
		printf("%s requires SM >= 2.0 for SurfaceWrites, exiting... \n", sSDKname);
		cutilDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
	}

	// load image from disk
    float* h_data = NULL;
    unsigned int width, height;
    char* image_path = cutFindFilePath(image_filename, argv[0]);
    if (image_path == NULL) {
        printf("Unable to source image input file: %s\n", image_filename);
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    cutilCheckError( cutLoadPGMf(image_path, &h_data, &width, &height));

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", image_filename, width, height);

    // load reference image from image (output)
    float *h_data_ref = (float*) malloc(size);
    char* ref_path = cutFindFilePath(ref_filename, argv[0]);
    if (ref_path == NULL) {
        printf("Unable to find reference image file: %s\n", ref_filename);
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    cutilCheckError( cutLoadPGMf(ref_path, &h_data_ref, &width, &height));

	// allocate device memory for result
    float* d_data = NULL;
    cutilSafeCall( cudaMalloc( (void**) &d_data, size));

    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    cutilSafeCall( cudaMallocArray( &cu_array, &channelDesc, width, height, cudaArraySurfaceLoadStore )); 

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

#if 1
    cutilSafeCall( cudaMemcpy( d_data, h_data, size, cudaMemcpyHostToDevice) );
    cutilSafeCall(cudaBindSurfaceToArray(output_surface, cu_array));

    surfaceWriteKernel<<< dimGrid, dimBlock >>>( d_data, width, height);
#else // this is what differs from the example simpleTexture
    cutilSafeCall( cudaMemcpyToArray( cu_array, 0, 0, h_data, size, cudaMemcpyHostToDevice));
#endif

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    cutilSafeCall( cudaBindTextureToArray( tex, cu_array, channelDesc));

    // warmup
    transformKernel<<< dimGrid, dimBlock, 0 >>>( d_data, width, height, angle);

    cutilSafeCall( cutilDeviceSynchronize() );
    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    // execute the kernel
    transformKernel<<< dimGrid, dimBlock, 0 >>>( d_data, width, height, angle);

    // check if kernel execution generated an error
    cutilCheckMsg("Kernel execution failed");

    cutilSafeCall( cutilDeviceSynchronize() );
    cutilCheckError( cutStopTimer( timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
    printf("%.2f Mpixels/sec\n", (width*height / (cutGetTimerValue( timer) / 1000.0f)) / 1e6);
    cutilCheckError( cutDeleteTimer( timer));

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( size);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_data, size, cudaMemcpyDeviceToHost) );

    // write result to file
    char output_filename[1024];
    strcpy(output_filename, "output.pgm");
    cutilCheckError( cutSavePGMf( "output.pgm", h_odata, width, height) );
    printf("Wrote '%s'\n", output_filename);

    // write regression file if necessary
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression"))
    {
        // write file for regression test
        cutilCheckError( cutWriteFilef( "./data/regression.dat", h_odata, width*height, 0.0));
    } 
    else 
    {
        // We need to reload the data from disk, because it is inverted upon output
        cutilCheckError( cutLoadPGMf(output_filename, &h_odata, &width, &height));

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", output_filename);
        printf("\treference: <%s>\n", ref_path);
        bTestResult = (bool)cutComparefe( h_odata, h_data_ref, width*height, MIN_EPSILON_ERROR );
    }

    cutilSafeCall(cudaFree(d_data));
    cutilSafeCall(cudaFreeArray(cu_array));
    cutFree(image_path);
    cutFree(ref_path);

    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (bTestResult ? QA_PASSED : QA_FAILED) );
}
