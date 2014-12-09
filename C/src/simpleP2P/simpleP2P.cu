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
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and 
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Shared Library Test Functions
#include <shrQATest.h>

#ifdef STRCASECMP
#undef STRCASECMP
#endif
#ifdef STRNCASECMP
#undef STRNCASECMP
#endif

#ifdef _WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

__global__ void SimpleKernel(float *src, float *dst)
{
    // Just a dummy kernel, doing enough for us to verify that everything
    // worked
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}

inline bool IsAppBuiltAs64()
{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
    return 1;
#else
    return 0;
#endif
}


int main(int argc, char **argv)
{
    shrQAStart(argc, argv);

    if (!IsAppBuiltAs64()) {
        printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
        exit(EXIT_SUCCESS);
    }

    // Number of GPUs
    printf("Checking for multiple GPUs...\n");
    int gpu_n;
    cutilSafeCall(cudaGetDeviceCount(&gpu_n));
    printf("CUDA-capable device count: %i\n", gpu_n);
    if (gpu_n < 2)
    {
        printf("Two or more Tesla(s) with (SM 2.0) class GPUs are required for %s.\n", argv[0]);
        printf("Waiving test.\n");
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
        exit(EXIT_SUCCESS);
    }

    // Query device properties
    cudaDeviceProp prop[64];
    int gpuid_tesla[64]; // we want to find the first two GPU's that can support P2P
    int gpu_count = 0;   // GPUs that meet the criteria

    for (int i=0; i < gpu_n; i++) {
        cutilSafeCall(cudaGetDeviceProperties(&prop[i], i));
        // Only Tesla boards based on Fermi can support P2P
        if ((!STRNCASECMP( prop[i].name, "Tesla",  5 )) 
           && (prop[i].major >= 2)
#ifdef _WIN32
            // on Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled
             && prop[i].tccDriver
#endif
            ) 
        {
            // This is an array of P2P capable GPUs
            gpuid_tesla[gpu_count++] = i;
        }
        printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
    }

    // Check for TCC for Windows
    if (gpu_count < 2)
    {
        printf("\nThis sample requires 2 Tesla GPUs to use P2P/UVA functionality.\n");
#ifdef _WIN32
        printf("\nFor Windows Vista/Win7, a TCC driver must be installed and enabled to use P2P/UVA functionality.\n");
#endif
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
        exit(EXIT_SUCCESS);
    }

#if CUDART_VERSION >= 4000
    // Check possibility for peer access
    printf("\nChecking GPU(s) for support of peer to peer memory access...\n");
    int can_access_peer_0_1, can_access_peer_1_0;
    // In this case we just pick the first two that we can support
    cutilSafeCall(cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_tesla[0], gpuid_tesla[1]));
    cutilSafeCall(cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_tesla[1], gpuid_tesla[0]));

    // Output results from P2P capabilities
    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid_tesla[0]].name, gpuid_tesla[0], 
                                                                 prop[gpuid_tesla[1]].name, gpuid_tesla[1] ,
                                                                 can_access_peer_0_1 ? "Yes" : "No");
    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid_tesla[1]].name, gpuid_tesla[1],
                                                                 prop[gpuid_tesla[0]].name, gpuid_tesla[0],
                                                                 can_access_peer_1_0 ? "Yes" : "No");

    if (can_access_peer_0_1 == 0 || can_access_peer_1_0 == 0)
    {
        printf("Two or more Tesla(s) with class GPUs are required for %s to run.\n", argv[0]);
        printf("Support for UVA requires a Tesla with SM 2.0 capabilities.\n");
        printf("Peer to Peer access is not available between GPU%d <-> GPU%d, waiving test.\n", gpuid_tesla[0], gpuid_tesla[1]);
        printf("PASSED\n");
        exit(EXIT_SUCCESS);
    }

    // Enable peer access
    printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid_tesla[0], gpuid_tesla[1]);
    cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
    cutilSafeCall(cudaDeviceEnablePeerAccess(gpuid_tesla[1], gpuid_tesla[0]));
    cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
    cutilSafeCall(cudaDeviceEnablePeerAccess(gpuid_tesla[0], gpuid_tesla[0]));

    // Check that we got UVA on both devices
    printf("Checking GPU%d and GPU%d for UVA capabilities...\n", gpuid_tesla[0], gpuid_tesla[1]);
    const bool has_uva = (prop[gpuid_tesla[0]].unifiedAddressing && prop[gpuid_tesla[1]].unifiedAddressing);

    printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid_tesla[0]].name, gpuid_tesla[0], (prop[gpuid_tesla[0]].unifiedAddressing ? "Yes" : "No") );
    printf("> %s (GPU%d) supports UVA: %s\n", prop[gpuid_tesla[1]].name, gpuid_tesla[1], (prop[gpuid_tesla[1]].unifiedAddressing ? "Yes" : "No") );

    if (has_uva) {
        printf("Both GPUs can support UVA, enabling...\n");
    } else {
        printf("At least one of the two GPUs does NOT support UVA, waiving test.\n");
        printf("PASSED\n");
        exit(EXIT_SUCCESS);
    }

    // Allocate buffers
    const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
    printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid_tesla[0], gpuid_tesla[1]);
    cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
    float* g0;
    cutilSafeCall(cudaMalloc(&g0, buf_size));
    cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
    float* g1;
    cutilSafeCall(cudaMalloc(&g1, buf_size));
    float* h0;
    cutilSafeCall(cudaMallocHost(&h0, buf_size)); // Automatically portable with UVA
        
    // Create CUDA event handles
    printf("Creating event handles...\n");
    cudaEvent_t start_event, stop_event;
    float time_memcpy;
    int eventflags = cudaEventBlockingSync;
    cutilSafeCall(cudaEventCreateWithFlags(&start_event, eventflags));
    cutilSafeCall(cudaEventCreateWithFlags(&stop_event, eventflags));

    // P2P memcopy() benchmark
    cutilSafeCall(cudaEventRecord(start_event, 0));
    for (int i=0; i<100; i++)
    {
        // With UVA we don't need to specify source and target devices, the
        // runtime figures this out by itself from the pointers
            
        // Ping-pong copy between GPUs
        if (i % 2 == 0)
            cutilSafeCall(cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault));
        else
            cutilSafeCall(cudaMemcpy(g0, g1, buf_size, cudaMemcpyDefault));
    }
    cutilSafeCall(cudaEventRecord(stop_event, 0));
    cutilSafeCall(cudaEventSynchronize(stop_event));
    cutilSafeCall(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
    printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n", gpuid_tesla[0], gpuid_tesla[1],
        (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);
 
    // Prepare host buffer and copy to GPU 0
    printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid_tesla[0]);
    for (int i=0; i<buf_size / sizeof(float); i++)
    {
        h0[i] = float(i % 4096);
    }
    cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
    cutilSafeCall(cudaMemcpy(g0, h0, buf_size, cudaMemcpyDefault));

    // Kernel launch configuration
    const dim3 threads(512, 1);
    const dim3 blocks((buf_size / sizeof(float)) / threads.x, 1);
 
    // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
    // output to the GPU 1 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n", 
            gpuid_tesla[1], gpuid_tesla[0], gpuid_tesla[1]);
    cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
    SimpleKernel<<<blocks, threads>>> (g0, g1);

    cutilSafeCall( cutilDeviceSynchronize() );

    // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
    // output to the GPU 0 buffer
    printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n", 
            gpuid_tesla[0], gpuid_tesla[1], gpuid_tesla[0]);
    cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
    SimpleKernel<<<blocks, threads>>> (g1, g0);

    cutilSafeCall( cutilDeviceSynchronize() );
 
    // Copy data back to host and verify
    printf("Copy data back to host from GPU%d and verify results...\n", gpuid_tesla[0]);
    cutilSafeCall(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDefault));
 
    int error_count = 0;
    for (int i=0; i<buf_size / sizeof(float); i++)
    {
        // Re-generate input data and apply 2x '* 2.0f' computation of both
        // kernel runs
        if (h0[i] != float(i % 4096) * 2.0f * 2.0f)
        {
            printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i], (float(i%4096)*2.0f*2.0f) );
            if (error_count++ > 10)
                break;
        }
    }
    // Disable peer access (also unregisters memory for non-UVA cases)
    printf("Enabling peer access...\n");
    cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
    cutilSafeCall(cudaDeviceDisablePeerAccess(gpuid_tesla[1]));
    cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
    cutilSafeCall(cudaDeviceDisablePeerAccess(gpuid_tesla[0]));

    // Cleanup and shutdown
    printf("Shutting down...\n");
    cutilSafeCall(cudaEventDestroy(start_event));
    cutilSafeCall(cudaEventDestroy(stop_event));
    cutilSafeCall(cudaSetDevice(gpuid_tesla[0]));
    cutilSafeCall(cudaFree(g0));
    cutilSafeCall(cudaSetDevice(gpuid_tesla[1]));
    cutilSafeCall(cudaFree(g1));
    cutilSafeCall(cudaFreeHost(h0));

    for( int i=0; i<gpu_n; i++ ) {	
        cutilSafeCall( cudaSetDevice(i) );
        cutilDeviceReset();
    }

    shrQAFinishExit(argc, (const char **)argv, (error_count == 0) ? QA_PASSED : QA_FAILED);

#else // Using CUDA 3.2 or older
    printf("simpleP2P requires CUDA 4.0 to build and run, waiving testing\n");
    shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
#endif

}

