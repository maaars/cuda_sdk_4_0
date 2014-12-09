/*
* Copyright 2008-2009 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/


#include "Exceptions.h"

#include <sstream>

namespace npp 
{

    Exception::Exception(const std::string & rMessage, 
                         const std::string & rFileName, 
                         unsigned int nLineNumber): sMessage_(rMessage)
                                                  , sFileName_(rFileName)
                                                  , nLineNumber_(nLineNumber)
    { ; }

    Exception::Exception(const Exception & rException): sMessage_(rException.sMessage_)
                                                      , sFileName_(rException.sFileName_)
                                                      , nLineNumber_(rException.nLineNumber_)
    { ; }

    Exception::~Exception()
    { ; }

    const
    std::string &
    Exception::message()
    const
    {
        return sMessage_;
    }

    const 
    std::string &
    Exception::fileName()
    const
    {
        return sFileName_;
    }

    unsigned int
    Exception::lineNumber()
    const
    {
        return nLineNumber_;
    }

    Exception * 
    Exception::clone()
    const
    {
        return new Exception(*this);
    }

    std::string
    Exception::toString()
    const
    {
        std::ostringstream oOutputString;
        
        oOutputString << fileName() << ":" << lineNumber() << ": " << message();
        
        return oOutputString.str();
    }


    std::ostream &
    operator << (std::ostream & rOutputStream, const Exception & rException)
    {
        rOutputStream << rException.toString();
        
        return rOutputStream;
    }

} // npp namespace