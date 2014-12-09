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
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the CUDA driver API and demonstrates how to use cuModuleLoad to load 
* the CUDA ptx (*.ptx) kernel just prior to kernel launch.
* 
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, CUDA
#include <cutil_inline.h>

// includes, project
#include <cutil_inline.h>
#include <shrQATest.h>

#include <iostream>
#include <cstring>

using namespace std;

const char *image_filename = "lena_bw.pgm";
const char *ref_filename   = "ref_rotated.pgm";
float angle = 0.5f;    // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

static CUresult initCUDA(int argc, char**argv, CUfunction*);

const char *sSDKsample = "simpleTextureDrv (Driver API)";

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

void
showHelp()
{
  printf("\n> [%s] Command line options\n", sSDKsample);
  printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "help") ) {
        showHelp();
        return 0;
    }

    runTest( argc, argv );
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    bool bTestResults = true;

    shrQAStart(argc, argv);
    
    // initialize CUDA
    CUfunction transform = NULL;
    if (initCUDA(argc, argv, &transform) != CUDA_SUCCESS) {
       shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }

    // load image from disk
    float* h_data = NULL;
    unsigned int width, height;
    char* image_path = cutFindFilePath(image_filename, argv[0]);
    if (image_path == NULL) {
        printf("Unable to find image file: '%s'\n", image_filename);
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    cutilCheckError( cutLoadPGMf(image_path, &h_data, &width, &height));

    size_t       size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", image_filename, width, height);

    // load reference image from image (output)
    float *h_data_ref = (float*) malloc(size);
    char* ref_path = cutFindFilePath(ref_filename, argv[0]);
    if (ref_path == NULL) {
        printf("Unable to find reference file %s\n", ref_filename);
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }
    cutilCheckError( cutLoadPGMf(ref_path, &h_data_ref, &width, &height));

    // allocate device memory for result
    CUdeviceptr d_data = (CUdeviceptr)NULL;
    cutilDrvSafeCall( cuMemAlloc( &d_data, size));

    // allocate array and copy image data
    CUarray cu_array;
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = width;
    desc.Height = height;
    cutilDrvSafeCall( cuArrayCreate( &cu_array, &desc ));
	CUDA_MEMCPY2D copyParam;
	memset(&copyParam, 0, sizeof(copyParam));
	copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copyParam.dstArray = cu_array;
	copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParam.srcHost = h_data;
	copyParam.srcPitch = width * sizeof(float);
	copyParam.WidthInBytes = copyParam.srcPitch;
	copyParam.Height = height;
    cutilDrvSafeCall(cuMemcpy2D(&copyParam));

    // set texture parameters
    CUtexref cu_texref;
    cutilDrvSafeCall(cuModuleGetTexRef(&cu_texref, cuModule, "tex"));
    cutilDrvSafeCall(cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT));
    cutilDrvSafeCall(cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_WRAP));
    cutilDrvSafeCall(cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_WRAP));
    cutilDrvSafeCall(cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR));
    cutilDrvSafeCall(cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES));
    cutilDrvSafeCall(cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_FLOAT, 1));

    cutilDrvSafeCall(cuParamSetTexRef(transform, CU_PARAM_TR_DEFAULT, cu_texref));

    // There are two ways to launch CUDA kernels via the Driver API.  
    // In this SDK sample, we illustrate both ways to pass parameters 
    // and specify parameters.  By default we use the simpler method.
    int block_size = 8;
    unsigned int timer = 0;
    if (1) {
    // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (simpler method)
        void *args[5] = { &d_data, &width, &height, &angle };

        cutilDrvSafeCallNoSync(cuLaunchKernel( transform, (width/block_size), (height/block_size), 1, 
                                                          block_size     , block_size     , 1, 
                                                          0,
                                                          NULL, args, NULL) );
        cutilDrvSafeCall( cuCtxSynchronize() );
        cutilCheckError( cutCreateTimer( &timer));
        cutilCheckError( cutStartTimer( timer));

        // launch kernel again for performance measurement
        cutilDrvSafeCallNoSync(cuLaunchKernel( transform, (width/block_size), (height/block_size), 1, 
                                                          block_size     , block_size     , 1, 
                                                          0,
                                                          NULL, args, NULL) );
    } else {
    // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (advanced method)
        int offset = 0;
        char argBuffer[256];

        // pass in launch parameters (not actually de-referencing CUdeviceptr).  CUdeviceptr is
        // storing the value of the parameters
        *((CUdeviceptr *)&argBuffer[offset]) = d_data;   offset += sizeof(d_data);
        *((unsigned int*)&argBuffer[offset]) = width;    offset += sizeof(width);
        *((unsigned int*)&argBuffer[offset]) = height;   offset += sizeof(height);
        *((float       *)&argBuffer[offset]) = angle;    offset += sizeof(angle);

        void *kernel_launch_config[5] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
            CU_LAUNCH_PARAM_BUFFER_SIZE,    &offset,
            CU_LAUNCH_PARAM_END
        };

        // new CUDA 4.0 Driver API Kernel launch call (warmup)
        cutilDrvSafeCallNoSync(cuLaunchKernel( transform, (width/block_size), (height/block_size), 1, 
                                                          block_size     , block_size     , 1, 
                                                          0,
                                                          NULL, NULL, (void **)&kernel_launch_config) );
        cutilDrvSafeCall( cuCtxSynchronize() );
        cutilCheckError( cutCreateTimer( &timer));
        cutilCheckError( cutStartTimer( timer));

        // launch kernel again for performance measurement
        cutilDrvSafeCallNoSync(cuLaunchKernel( transform, (width/block_size), (height/block_size), 1, 
                                                          block_size     , block_size     , 1, 
                                                          0, 0,
                                                          NULL, (void **)&kernel_launch_config) );
    }


    cutilDrvSafeCall( cuCtxSynchronize() );
    cutilCheckError( cutStopTimer( timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
    printf("%.2f Mpixels/sec\n", (width*height / (cutGetTimerValue( timer) / 1000.0f)) / 1e6);
    cutilCheckError( cutDeleteTimer( timer));

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( size);
    // copy result from device to host
    cutilDrvSafeCall( cuMemcpyDtoH( h_odata, d_data, size) );

    // write result to file
    char output_filename[1024];
    strcpy(output_filename, image_path);
    strcpy(output_filename + strlen(image_path) - 4, "_out.pgm");
    cutilCheckError( cutSavePGMf( output_filename, h_odata, width, height));
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
        bTestResults = cutComparefe( h_odata, h_data_ref, width*height, MIN_EPSILON_ERROR );
    }

    // cleanup memory
    cutilDrvSafeCall(cuMemFree(d_data));
    cutilDrvSafeCall(cuArrayDestroy(cu_array));

    cutFree(image_path);
    cutFree(ref_path);

    cutilDrvSafeCall(cuCtxDetach(cuContext));

    shrQAFinishExit(argc, (const char **)argv, bTestResults ? QA_PASSED : QA_FAILED);
}

bool inline
findModulePath(const char *module_file, string & module_path, char **argv, string & ptx_source)
{
    char *actual_path = cutFindFilePath(module_file, argv[0]);
    if (actual_path) {
       module_path = actual_path;
    } else {
       printf("> findModulePath file not found: <%s> \n", module_file); 
       return false;
    }

    if (module_path.empty()) {
       printf("> findModulePath file not found: <%s> \n", module_file); 
       return false;
    } else {
       printf("> findModulePath <%s>\n", module_path.c_str());

	   if (module_path.rfind(".ptx") != string::npos) {
		   FILE *fp = fopen(module_path.c_str(), "rb");
		   fseek(fp, 0, SEEK_END);
		   int file_size = ftell(fp);
           char *buf = new char[file_size+1];
           fseek(fp, 0, SEEK_SET);
           fread(buf, sizeof(char), file_size, fp);
           fclose(fp);
           buf[file_size] = '\0';
           ptx_source = buf;
           delete[] buf;
	   }
	   return true;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! This initializes CUDA, and loads the *.ptx CUDA module containing the
//! kernel function.  After the module is loaded, cuModuleGetFunction 
//! retrieves the CUDA function pointer "cuFunction" 
////////////////////////////////////////////////////////////////////////////////
static CUresult
initCUDA(int argc, char **argv, CUfunction* transform)
{
    CUfunction cuFunction = 0;
    CUresult status;
    int major = 0, minor = 0, devID = 0;
    char deviceName[100];
    string module_path, ptx_source;

    cuDevice = cutilChooseCudaDeviceDrv(argc, argv, &devID);

    // get compute capabilities and the devicename
    cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, cuDevice) );
    cutilDrvSafeCallNoSync( cuDeviceGetName(deviceName, 256, cuDevice) );
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    status = cuCtxCreate( &cuContext, 0, cuDevice );
    if ( CUDA_SUCCESS != status ) {
        printf("cuCtxCreate(0) returned %d\n-> %s\n", status, getCudaDrvErrorString(status));
        goto Error;
    }

    // first search for the module_path before we try to load the results
    if (!findModulePath ("simpleTexture_kernel.ptx", module_path, argv, ptx_source)) {
       if (!findModulePath ("simpleTexture_kernel.cubin", module_path, argv, ptx_source)) {
          printf("> findModulePath could not find <simpleTexture_kernel> ptx or cubin\n");
          status = CUDA_ERROR_NOT_FOUND;
          goto Error;
       }
    } else {
       printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

	if (module_path.rfind("ptx") != string::npos) {
		// in this branch we use compilation with parameters
		const unsigned int jitNumOptions = 3;
		CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
		void **jitOptVals = new void*[jitNumOptions];

		// set up size of compilation log buffer
		jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		int jitLogBufferSize = 1024;
		jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

		// set up pointer to the compilation log buffer
		jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
		char *jitLogBuffer = new char[jitLogBufferSize];
		jitOptVals[1] = jitLogBuffer;

		// set up pointer to set the Maximum # of registers for a particular kernel
		jitOptions[2] = CU_JIT_MAX_REGISTERS;
		int jitRegCount = 32;
		jitOptVals[2] = (void *)(size_t)jitRegCount;

		status = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);

		printf("> PTX JIT log:\n%s\n", jitLogBuffer);
	} else {
		status = cuModuleLoad(&cuModule, module_path.c_str());
	}

	if ( CUDA_SUCCESS != status ) {
        goto Error;
    }

    status = cuModuleGetFunction( &cuFunction, cuModule, "transformKernel" );
    if ( CUDA_SUCCESS != status )
        goto Error;
    *transform = cuFunction;
    return CUDA_SUCCESS;
Error:
    cuCtxDetach(cuContext);
    return status;
}
