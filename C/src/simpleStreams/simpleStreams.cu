/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 *
 *
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel is used to 
 * initialize an array to a specific value, after which the array is 
 * copied to the host (CPU) memory.  To increase performance, multiple 
 * kernel/memcopy pairs are launched asynchronously, each pair in its 
 * own stream.  Devices with Compute Capability 1.1 can overlap a kernel
 * and a memcopy as long as they are issued in different streams.  Kernels
 * are serialized.  Thus, if n pairs are launched, streamed approach
 * can reduce the memcopy cost to the (1/n)th of a single copy of the entire
 * data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5 
 * microsecond precision.
 *
 * Elapsed times are averaged over nreps repetitions (10 by default).
 *
*/

const char *sSDKsample = "simpleStreams";

const char *sEventSyncMethod[] = 
{ 
	"cudaEventDefault", 
	"cudaEventBlockingSync", 
	"cudaEventDisableTiming", 
	NULL 
};

const char *sDeviceSyncMethod[] = 
{ 
	"cudaDeviceScheduleAuto", 
	"cudaDeviceScheduleSpin", 
	"cudaDeviceScheduleYield", 
	"INVALID", 
	"cudaDeviceScheduleBlockingSync", 
	NULL 
};

// CUDA utilities and system includes
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

// Include headers
#include <stdio.h>

// Shared Library Test Functions
#include <shrUtils.h>
#include <shrQATest.h>

#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

__global__ void init_array(int *g_data, int *factor, int num_iterations)
{ 
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0;i<num_iterations;i++)
	    g_data[idx] += *factor;	// non-coalesced on purpose, to burn time
}

int correct_data(int *a, const int n, const int c)
{
    for(int i = 0; i < n; i++) {
        if(a[i] != c) {
           printf("%d: %d %d\n", i, a[i], c);
           return 0;
        }
    }
    return 1;
}

inline void 
AllocateHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
    if (bPinGenericMemory)
    {
        // allocate a generic page-aligned chunk of system memory
    #ifdef WIN32
        printf("> VirtualAlloc() allocating %4.2f Mbytes of (generic page-aligned system memory)\n", (float)nbytes/1048576.0f);
        *pp_a = (int *) VirtualAlloc( NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE );
    #else
        printf("> mmap() allocating %4.2f Mbytes (generic page-aligned system memory)\n", (float)nbytes/1048576.0f);
        *pp_a = (int *) mmap        ( NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0 );
    #endif

        *ppAligned_a = (int *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);

        printf("> cudaHostRegister() registering %4.2f Mbytes of generic allocated system memory\n", (float)nbytes/1048576.0f);
        // pin allocate memory
        cutilSafeCall( cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped) );
    }
    else
#endif
    {
        printf("> cudaMallocHost() allocating %4.2f Mbytes of system memory\n", (float)nbytes/1048576.0f);
        // allocate host memory (pinned is required for achieve asynchronicity)
        cutilSafeCall( cudaMallocHost((void**)pp_a, nbytes) ); 
        *ppAligned_a = *pp_a; 
    }
}

inline void
FreeHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
    // CUDA 4.0 support pinning of generic host memory
    if (bPinGenericMemory)
    {
        // unpin and delete host memory
        cutilSafeCall( cudaHostUnregister(*ppAligned_a) );
  #ifdef WIN32
        VirtualFree(*pp_a, 0, MEM_RELEASE);
  #else
        munmap(*pp_a, nbytes);
  #endif
    }
    else
#endif
    {
        cudaFreeHost(*pp_a);
    }
}

void printHelp()
{
    printf("Usage: %s [options below]\n", sSDKsample);
    printf("\t--sync_method for CPU/GPU synchronization\n");
    printf("\t             (0=Automatic Blocking Scheduling)\n");
    printf("\t             (1=Spin Blocking Scheduling)\n");
    printf("\t             (2=Yield Blocking Scheduling)\n");
    printf("\t   <Default> (4=Blocking Sync Event Scheduling for low CPU utilization)\n");
    printf("\t--use_generic_memory (default) use generic page-aligned for system memory\n");
    printf("\t--use_cuda_malloc_host (optional) use cudaMallocHost to allocate system memory\n");
}

#if defined(__APPLE__) || defined(MACOSX)
#define DEFAULT_PINNED_GENERIC_MEMORY false
#else
#define DEFAULT_PINNED_GENERIC_MEMORY true
#endif

int main(int argc, char *argv[])
{
    int cuda_device = 0;
    int nstreams = 4;               // number of streams for CUDA calls
    int nreps = 10;                 // number of times each experiment is repeated
    int n = 16 * 1024 * 1024;       // number of ints in the data set
    int nbytes = n * sizeof(int);   // number of data bytes
    dim3 threads, blocks;           // kernel launch configuration
    float elapsed_time, time_memcpy, time_kernel;   // timing variables
    float scale_factor = 1.0f;

    // allocate generic memory and pin it laster instead of using cudaHostAlloc()

    bool bPinGenericMemory  = DEFAULT_PINNED_GENERIC_MEMORY; // we want this to be the default behavior
    int  device_sync_method = cudaDeviceBlockingSync; // by default we use BlockingSync

    int niterations;	// number of iterations for the loop inside the kernel

    shrQAStart(argc, argv);

    printf("[ %s ]\n\n", sSDKsample);
    if( cutCheckCmdLineFlag( argc, (const char **)argv, "help") ) {
        printHelp();
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
    }
    if( cutGetCmdLineArgumenti( argc, (const char **)argv, "sync_method", &device_sync_method) ) {
        if (device_sync_method == 0 || device_sync_method == 1 || device_sync_method == 2 || device_sync_method == 4) {
            printf("Specifying device_sync_method = %d, setting reps to 100 to demonstrate steady state\n", device_sync_method);
            nreps = 100;
        } else {
            printf("Invalid command line option sync_method=\"%d\"\n", device_sync_method);
            shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
        }
    }
    if( cutCheckCmdLineFlag( argc, (const char **)argv, "use_generic_memory") ) {
#if defined(__APPLE__) || defined(MACOSX)
        bPinGenericMemory = false;  // Generic Pinning of System Paged memory not currently supported on Mac OSX 
#else
        bPinGenericMemory = true;
#endif
    }
    if( cutCheckCmdLineFlag( argc, (const char **)argv, "use_cuda_malloc_host") ) {
        bPinGenericMemory = false;
    }

    printf("\n> ");
    cuda_device = cutilChooseCudaDevice(argc, argv);

    // check the compute capability of the device
    int num_devices=0;
    cutilSafeCall( cudaGetDeviceCount(&num_devices) );
    if(0==num_devices)
    {
        printf("your system does not have a CUDA capable device, waiving test...\n");
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
    }
	
    // check if the command-line chosen device ID is within range, exit if not
    if( cuda_device >= num_devices )
    {
        printf("cuda_device=%d is invalid, must choose device ID between 0 and %d\n", cuda_device, num_devices-1);
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
    }

    cudaSetDevice( cuda_device );

    // Checking for compute capabilities
    cudaDeviceProp deviceProp;
    cutilSafeCall( cudaGetDeviceProperties(&deviceProp, cuda_device) );
    if( (1 == deviceProp.major) && (deviceProp.minor < 1)) {
        printf("%s does not have Compute Capability 1.1 or newer.  Reducing workload.\n", deviceProp.name);
    }

    if(deviceProp.major >= 2) {
        niterations = 100;
    } else {		    
        if(deviceProp.minor > 1) {
            niterations = 5;
        } else {
            niterations = 1; // reduced workload for compute capability 1.0 and 1.1
        }
    }

    // Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
    if (bPinGenericMemory) {
       printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");
       if (deviceProp.canMapHostMemory == 0) {
          printf("Using cudaMallocHost, CUDA device does not support mapping of generic host memory\n");
          bPinGenericMemory = false;
       }
    }

#ifdef __DEVICE_EMULATION__
    n = 4096;   // reduced workload for emulation (n should be divisible by 512*nstreams)
#else
    // Anything that is less than 32 Cores will have scaled down workload
    scale_factor = max((32.0f / (ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount) ), 1.0f);
    n = (int)rint( (float)n / scale_factor );
#endif

    printf("> CUDA Capable: SM %d.%d hardware\n", deviceProp.major, deviceProp.minor);
    printf("> %d Multiprocessor(s) x %d (Cores/Multiprocessor) = %d (Cores)\n", 
            deviceProp.multiProcessorCount,
            ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    printf("> scale_factor = %1.4f\n", 1.0f/scale_factor);
    printf("> array_size   = %d\n\n", n);

    // enable use of blocking sync, to reduce CPU usage
    printf("> Using CPU/GPU Device Synchronization method (%s)\n", sDeviceSyncMethod[device_sync_method]);
    cudaSetDeviceFlags( device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0 ) );

    // allocate host memory
    int c = 5;                      // value to which the array will be initialized
    int *h_a = 0;                   // pointer to the array data in host memory
    int *hAligned_a = 0;           // pointer to the array data in host memory (aligned to MEMORY_ALIGNMENT)

    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    AllocateHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    // allocate device memory
    int *d_a = 0, *d_c = 0;             // pointers to data and init value in the device memory
    cutilSafeCall( cudaMalloc((void**)&d_a, nbytes) );
    cutilSafeCall( cudaMalloc((void**)&d_c, sizeof(int)) );
    cutilSafeCall( cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice) );

    printf("\nStarting Test\n");

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++) {
        cutilSafeCall( cudaStreamCreate(&(streams[i])) );
    }

    // create CUDA event handles
    // use blocking sync
    cudaEvent_t start_event, stop_event;
    int eventflags = ( (device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync: cudaEventDefault );

    cutilSafeCall( cudaEventCreateWithFlags(&start_event, eventflags) );
    cutilSafeCall( cudaEventCreateWithFlags(&stop_event, eventflags) );

    // time memcopy from device
    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
    cudaMemcpyAsync(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cutilSafeCall( cudaEventElapsedTime(&time_memcpy, start_event, stop_event) );
    printf("memcopy:\t%.2f\n", time_memcpy);
    
    // time kernel
    threads=dim3(512, 1);
    blocks=dim3(n / threads.x, 1);
    cudaEventRecord(start_event, 0);
    init_array<<<blocks, threads, 0, streams[0]>>>(d_a, d_c, niterations);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&time_kernel, start_event, stop_event) );
    printf("kernel:\t\t%.2f\n", time_kernel);

    //////////////////////////////////////////////////////////////////////
    // time non-streamed execution for reference
    threads=dim3(512, 1);
    blocks=dim3(n / threads.x, 1);
    cudaEventRecord(start_event, 0);
    for(int k = 0; k < nreps; k++)
    {
        init_array<<<blocks, threads>>>(d_a, d_c, niterations);
        cudaMemcpy(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("non-streamed:\t%.2f (%.2f expected)\n", elapsed_time / nreps, time_kernel + time_memcpy);

    //////////////////////////////////////////////////////////////////////
    // time execution with nstreams streams
    threads=dim3(512,1);
    blocks=dim3(n/(nstreams*threads.x),1);
    memset(hAligned_a, 255, nbytes);     // set host memory bits to all 1s, for testing correctness
    cudaMemset(d_a, 0, nbytes); // set device memory to all 0s, for testing correctness
    cudaEventRecord(start_event, 0);
    for(int k = 0; k < nreps; k++)
    {
        // asynchronously launch nstreams kernels, each operating on its own portion of data
        for(int i = 0; i < nstreams; i++)
            init_array<<<blocks, threads, 0, streams[i]>>>(d_a + i * n / nstreams, d_c, niterations);

        // asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
        //   commence executing when all previous CUDA calls in stream x have completed
        for(int i = 0; i < nstreams; i++)
            cudaMemcpyAsync(hAligned_a + i * n / nstreams, d_a + i * n / nstreams, nbytes / nstreams, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cutilSafeCall( cudaEventElapsedTime(&elapsed_time, start_event, stop_event) );
    printf("%d streams:\t%.2f (%.2f expected with compute capability 1.1 or later)\n", nstreams, elapsed_time / nreps, time_kernel + time_memcpy / nstreams);

    // check whether the output is correct
    printf("-------------------------------\n");
    bool bResults = correct_data(hAligned_a, n, c*nreps*niterations);

    // release resources
    for(int i = 0; i < nstreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
    FreeHostMemory( bPinGenericMemory, &h_a, &hAligned_a, nbytes );

    cudaFree(d_a);
    cudaFree(d_c);

    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, bResults ? QA_PASSED : QA_FAILED);
}
