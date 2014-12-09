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

typedef int mutex_t;
enum MUTEX_STATE {UNLOCKED=0, LOCKED};


#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
typedef unsigned long long int pointer_int_t;
#else
typedef unsigned int pointer_int_t;
#endif

// Attention: works only when first thread of warp is active and all threads of warp exit critical section together (no divergent returns)
class ScopedWarpLock {

public:
	__device__
	ScopedWarpLock(mutex_t* mutex) : m_mutex(mutex)  {
		int warpThreadIdx = threadIdx.x & 31;

		// Poll until lock is released
		while( (warpThreadIdx == 0) && atomicCAS(mutex, UNLOCKED, LOCKED) == LOCKED );
	}

	__device__
	~ScopedWarpLock() {
		*m_mutex = UNLOCKED;
	}

private:
	mutex_t *m_mutex;
};

template<class T>
class Container {

public:
	__device__
	Container() : m_mutex(UNLOCKED) {;}

	__device__
	virtual void push(T e) = 0;

	__device__
	virtual bool pop(T &e) = 0;

protected:
	mutex_t m_mutex;
};


template<class T>
class Vector : public Container<T> {

public:
	__device__
	Vector(int max_size) :  m_top(-1) {
		m_data = new T[max_size];
	}

	__device__
	~Vector() {
		delete m_data;
	}

	__device__
	virtual
	void push(T e) {
		int idx = atomicAdd(&(this->m_top), 1);
		m_data[idx+1] = e;
	}

	__device__
	virtual
	bool pop(T &e) {
		if( m_top >= 0 ) {
			int idx = atomicAdd( &(this->m_top), -1 );
			if( idx >= 0 ) {
				e = m_data[idx];
				return true;
			}
		}
		return false;
		
	}


private:
	int m_size;
	T* m_data;

	int m_top;
};

template<class T>
class SingleLinkElement {

public:
	__device__
	SingleLinkElement(T e) : m_data(e), m_next(0) {
	}

	__device__
	void setNext(SingleLinkElement<T>* next) {
		m_next = next;
	}

	__device__
	SingleLinkElement<T>* getNext() {
		return m_next;
	}

	__device__
	T operator() (){
		return m_data;
	}

private:
	T m_data;
	SingleLinkElement<T>* m_next;
};


template<class T>
class StackAtomicPush : public Container<T> {

public:
	__device__
	StackAtomicPush() {
		m_top = 0;
	}

	__device__
	virtual 	
	void push(T e) {
		SingleLinkElement<T>* newElement = new SingleLinkElement<T>(e);

		SingleLinkElement<T>* old_top = (SingleLinkElement<T>*) atomicExch( (pointer_int_t*)&(this->m_top), 
                                                                                    (pointer_int_t)newElement );
		newElement->setNext(old_top);
	}

protected:
	SingleLinkElement<T>* m_top;

};

template<class T>
class StackAtomicPop : public StackAtomicPush<T> {

public:
	__device__
	StackAtomicPop<T>() {;}

	__device__
	virtual
	bool pop(T &e) {
		SingleLinkElement<T>* curr_top = this->m_top;
		if( curr_top == 0 ) return false;

		SingleLinkElement<T>* next = curr_top->getNext();
		
		SingleLinkElement<T>* old_top = (SingleLinkElement<T>*) atomicCAS( (pointer_int_t*)&(this->m_top), 
                                                                                   (pointer_int_t)curr_top, 
                                                                                   (pointer_int_t)next );
		
		// Check for concurrent modifications and possibly repeat 
		while( old_top != curr_top ) {
			curr_top = old_top;
			if( curr_top == 0 ) return false;

			next = curr_top->getNext();
		
			old_top = (SingleLinkElement<T>*) atomicCAS( (pointer_int_t*)&(this->m_top), 
                                                                     (pointer_int_t)curr_top, 
                                                                     (pointer_int_t)next );
		}
		
		if( old_top != 0 ) {
			e = (*old_top)();
			delete old_top;
			return true;
		} else {
			return false;
		}
	}

	
};


template<class T>
class StackWarpLockPop : public StackAtomicPush<T> {

public:
	__device__
	StackWarpLockPop<T>() {;}

	__device__
	virtual
	bool pop(T &e) {

		__shared__ SingleLinkElement<T>* next;

		SingleLinkElement<T>* old_top;

		int threadWarpIdx = threadIdx.x & 31;

		{
			ScopedWarpLock lock(&(this->m_mutex));

			if( threadWarpIdx == 0 ) {
				next = this->m_top;
			}

			// Serialize
			for( int i=0; i<32; ++i ) {
				if( threadWarpIdx == i ) {
					old_top = next;

					if( old_top != 0 )
						next = old_top->getNext();
				}
			}

			if( threadWarpIdx == 31 ) {
				this->m_top = next;	
				__threadfence();
			}
		}
		
		if( old_top != 0 ) {
			e = (*old_top)();
			delete old_top;
			return true;
		} else {
			return false;
		}
	}

};
