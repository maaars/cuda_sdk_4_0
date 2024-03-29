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

///////////////////////////////////////////////////////////////////////////////
// Monte Carlo: Estimate Pi
// ========================
//
// This sample demonstrates a very simple Monte Carlo estimation for Pi.
//
// This file, main.cpp, contains the setup information to run the test, for
// example parsing the command line and integrating this sample with the
// samples framework. As such it is perhaps less interesting than the guts of
// the sample. Readers wishing to skip the clutter are advised to skip straight
// to Test.operator() in test.cpp.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <shrUtils.h>
#include <shrQATest.h>
#include <cutil_inline.h>

#include "../inc/test.h"

// SDK information
static const char *shrLogFile = "MonteCarloEstimatePiQ.txt";

// Forward declarations
void showHelp(const int argc, const char **argv);
template <typename Real>
void runTest(int argc, const char **argv);

int main(int argc, char **argv)
{
    using std::invalid_argument;
    using std::string;

    // Open the log file
	shrQAStart(argc, argv);
    shrSetLogFileName(shrLogFile);
    shrLog("Monte Carlo Estimate Pi (with batch QRNG)\n");
    shrLog("=========================================\n\n");

    // If help flag is set, display help and exit immediately
    if (shrCheckCmdLineFlag(argc, (const char **)argv, "help"))
    {
        shrLog("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
    }

    // Check the precision (checked against the device capability later)
    try
    {
        char *value;
        if (shrGetCmdLineArgumentstr(argc, (const char **)argv, "precision", &value))
        {
            // Check requested precision is valid
            string prec(value);
            if (prec.compare("single") == 0 || prec.compare("\"single\"") == 0)
                runTest<float>(argc, (const char **)argv);
            else if (prec.compare("double") == 0 || prec.compare("\"double\"") == 0)
            {
                runTest<double>(argc, (const char **)argv);
            }
            else
            {
                shrLogEx(LOGBOTH | ERRORMSG, 0, "specified precision (%s) is invalid, must be \"single\".\n", value);
                throw invalid_argument("precision");
            }
        }
        else
        {
            runTest<float>(argc, (const char **)argv);
        }
    }
    catch (invalid_argument &e)
    {
        shrLogEx(LOGBOTH | ERRORMSG, 0, "invalid command line argument (%s)\n", e.what());
        shrCheckErrorEX(false, true, NULL);
    }
    
    // Finish
    cutilDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}

template <typename Real>
void runTest(int argc, const char **argv)
{
    using std::invalid_argument;
    using std::runtime_error;
    
    try
    {
        Test<Real> test;
        int deviceCount = 0;
        cudaError_t cudaResult  = cudaSuccess;

        // by default specify GPU Device == 0
        test.device             = 0;

        // Get number of available devices
        cudaResult = cudaGetDeviceCount(&deviceCount);
        if (cudaResult != cudaSuccess)
        {
            shrLogEx(LOGBOTH | ERRORMSG, 0, "could not get device count.\n");
            throw runtime_error("cudaGetDeviceCount");
        }

        // Parse command line
        if (shrCheckCmdLineFlag(argc, argv, "qatest"))
        {
            test.numSims         = k_sims_qa;
            test.threadBlockSize = k_bsize_qa;
        }

        {
            char *value = 0;
            if (shrGetCmdLineArgumentstr(argc, argv, "device", &value))
            {
                test.device = (int)atoi(value);
                if (test.device >= deviceCount)
                {
                    shrLogEx(LOGBOTH | ERRORMSG, 0, "invalid target device specified on command line (device %d does not exist).\n", test.device);
                    throw invalid_argument("device");
                }
            }
            else
            {
                test.device = cutGetMaxGflopsDeviceId();
            }
            if (shrGetCmdLineArgumentstr(argc, argv, "sims", &value))
            {
                test.numSims = (unsigned int)atoi(value);
                if (test.numSims < k_sims_min || test.numSims > k_sims_max)
                {
                    shrLogEx(LOGBOTH | ERRORMSG, 0, "specified number of simulations (%d) is invalid, must be between %d and %d.\n", test.numSims, k_sims_min, k_sims_max);
                    throw invalid_argument("sims");
                }
            }
            else
            {
                test.numSims = k_sims_def;
            }
            if (shrGetCmdLineArgumentstr(argc, argv, "block-size", &value))
            {
                // Determine max threads per block
                cudaDeviceProp deviceProperties;
                cudaResult = cudaGetDeviceProperties(&deviceProperties, test.device);
                if (cudaResult != cudaSuccess)
                {
                    shrLogEx(LOGBOTH | ERRORMSG, 0, "cound not get device properties for device %d.\n", test.device);
                    throw runtime_error("cudaGetDeviceProperties");
                }
                // Check requested size is valid
                test.threadBlockSize = (unsigned int)atoi(value);
                if (test.threadBlockSize < k_bsize_min || test.threadBlockSize > static_cast<unsigned int>(deviceProperties.maxThreadsPerBlock))
                {
                    shrLogEx(LOGBOTH | ERRORMSG, 0, "specified block size (%d) is invalid, must be between %d and %d for device %d.\n", test.threadBlockSize, k_bsize_min, deviceProperties.maxThreadsPerBlock, test.device);
                    throw invalid_argument("block-size");
                }
            }
            else
            {
                test.threadBlockSize = k_bsize_def;
            }
        }

        // Execute
        shrCheckErrorEX(test(), true, NULL);
    }
    catch (invalid_argument &e)
    {
        shrLogEx(LOGBOTH | ERRORMSG, 0, "invalid command line argument (%s)\n", e.what());
        shrCheckErrorEX(false, true, NULL);
    }
    catch (runtime_error &e)
    {
        shrLogEx(LOGBOTH | ERRORMSG, 0, "runtime error (%s)\n", e.what());
        shrCheckErrorEX(false, true, NULL);
    }
}

void showHelp(int argc, const char **argv)
{
    using std::cout;
    using std::endl;
    using std::left;
    using std::setw;

    if (argc > 0)
        cout << endl << argv[0] << endl;
    cout << endl << "Syntax:" << endl;
    cout << left;
    cout << "    " << setw(20) << "--device=<device>" << "Specify device to use for execution" << endl;
    cout << "    " << setw(20) << "--sims=<N>" << "Specify number of Monte Carlo simulations" << endl;
    cout << "    " << setw(20) << "--block-size=<N>" << "Specify number of threads per block" << endl;
    cout << "    " << setw(20) << "--precision=<P>" << "Specify the precision (\"single\")" << endl;
    cout << endl;
    cout << "    " << setw(20) << "--noprompt" << "Skip prompt before exit" << endl;
    cout << endl;
}
