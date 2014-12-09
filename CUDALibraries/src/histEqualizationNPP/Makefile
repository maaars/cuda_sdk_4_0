################################################################################
#
# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################

CUDA_INSTALL_PATH ?= /usr/local/cuda

ifdef cuda-install
	CUDA_INSTALL_PATH := $(cuda-install)
endif

SDK_INSTALL_PATH ?= ../..

# Detect OS the being used
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
endif

ifeq ($(dbg),1)
   BINDIR ?= ../../bin/$(OSLOWER)/debug
else
   BINDIR ?= ../../bin/$(OSLOWER)/release
endif

# Autodetect 32-bit or 64-bit OS Architecture
HP_64 = $(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

ifneq ($(HP_64),)
   CUDALIBARCH := 64
   LIBARCH     := 64
   LIB_ARCH    := x86_64
else
   CUDALIBARCH :=
   LIBARCH     := 32
   LIB_ARCH    := i386
endif
ifeq ($(x86_64),1)
   CUDALIBARCH := 64
   LIBARCH     := 64
   LIB_ARCH    := x86_64
endif
ifeq ($(i386),1)
   CUDALIBARCH := 
   LIBARCH     := 32
   LIB_ARCH    := i386
endif
FREEIMAGELIBARCH := $(LIBARCH)

ifneq ($(DARWIN),)
   ifneq ($(SNOWLEOPARD),)
      OS_VERSION   := 10_6
   else
      OS_VERSION   := 10_5
   endif
endif
ifneq ($(DARWIN),)
   CUDALIBARCH :=
   ifeq ($(x86_64),1)
      CXX_ARCH_FLAGS += -arch x86_64 -m64
   else
      ifeq ($(i386),1)
         CXX_ARCH_FLAGS += -arch i386 -m32
      else
         ifneq ($(HP_64),)
            CXX_ARCH_FLAGS += -arch x86_64 -m64
         else
            CXX_ARCH_FLAGS += -arch i386 -m32
         endif
      endif
   endif
   FREEIMAGELIBARCH := _$(OS_VERSION)
endif

CXX := g++ -fPIC $(CXX_ARCH_FLAGS) 
INC := -I. -I../../common/UtilNPP/ -I../../../shared/inc -I$(CUDA_INSTALL_PATH)/include/

LIB := -L$(SDK_INSTALL_PATH)/common/lib/ -lnpp -L$(CUDA_INSTALL_PATH)/lib$(CUDALIBARCH)/ -L$(CUDA_INSTALL_PATH)/lib/ -lcudart
LIB += -L$(SDK_INSTALL_PATH)/common/FreeImage/lib/$(OSLOWER)


ifneq ($(DARWIN),)
   LIB += -Xlinker -rpath $(SDK_INSTALL_PATH)/common/lib/ -Xlinker -rpath $(CUDA_INSTALL_PATH)/lib/
endif


all:
	$(CXX) $(INC) $(LIB) -o histEqualizationNPP histEqualizationNPP.cpp -lUtilNPP_$(LIB_ARCH) -lfreeimage$(FREEIMAGELIBARCH)
	mkdir -p ../../bin
	mkdir -p ../../bin/$(OSLOWER)
	mkdir -p ../../bin/$(OSLOWER)/release
	mkdir -p ../../bin/$(OSLOWER)/debug
	cp histEqualizationNPP $(BINDIR)

clean:
	@rm -f histEqualizationNPP $(BINDIR)/histEqualizationNPP

clobber:
	@rm -f histEqualizationNPP $(BINDIR)/histEqualizationNPP
