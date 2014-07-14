#ifndef FREE_LIST_HH_INCLUDED
#define FREE_LIST_HH_INCLUDED

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <atomic>
#include <mutex>
#include <condition_variable>

using std::atomic;
using std::mutex;
using std::condition_variable;
using std::unique_lock;

// This file implements three free-list allocators.  These are not
// real allocators, in the sense that they do not manage memory
// allocations.  Instead, they are simple lists of already allocated
// but available objects.  All implementation take a template type T.
// The requirement on the template type is that it defines next
// pointer (class T { T * next; ... })
// 
template <class T> 
class free_list_dynamic_allocator {
	atomic<T *> head;

public:
	free_list_dynamic_allocator() : head(0) {};

	void preallocate(unsigned int n) {
		while(n-- > 0) {
			T * new_head = new T();
			new_head->next = head;
			head = new_head;
		}
	}

	T * allocate() {
		T * obj;
		do {
			obj = head;
			if (!obj) 
				return new T();
		} while (!atomic_compare_exchange_weak(&head, &obj, obj->next));

		return obj;
	}

	void recycle(T * obj) {
		do {
			obj->next = head;
		} while(!atomic_compare_exchange_weak(&head, &(obj->next), obj));
	}
};

template <class T> 
class free_list_static_spinning_allocator {
	// we allocate and recycle these handles.  This is the list of
	// handles that are available for use by the front-end.
	// 
	atomic<T *> head;

public:

	free_list_static_spinning_allocator() : head(0) {};
	free_list_static_spinning_allocator(T * begin, T * end) 
		: head(0) {
		while(begin != end) {
			begin->next = head;
			head = begin;
			++begin;
		}
	}

	T * allocate() {
		T * obj;
		do {
			obj = head;
			if (!obj) 
				continue;
		} while (!atomic_compare_exchange_weak(&head, &obj, obj->next));

		return obj;
	}

	void recycle(T * obj) {
		do {
			obj->next = head;
		} while(!atomic_compare_exchange_weak(&head, &(obj->next), obj));
	}
};

template <class T> 
class free_list_static_allocator {
	// we allocate and recycle these handles.  This is the list of
	// handles that are available for use by the front-end.
	// 
	atomic<T *> head;

	mutex mtx;
	condition_variable cv;

public:

	free_list_static_allocator() : head(0) {};
	free_list_static_allocator(T * begin, T * end) {
		preallocate(begin, end);
	}

	void preallocate(T * begin, T * end) {
		for(head = 0; begin != end; ++begin) {
			begin->next = head;
			head = begin;
			++begin;
		}
	}

	T * allocate() {
		unique_lock<std::mutex> lock(mtx);
		while(!head)
			cv.wait(lock);

		T* obj = head;
		head = obj->next;

		return obj;
	}

	void recycle(T * obj) {
		unique_lock<mutex> lock(mtx);
		obj->next = head;
		head = obj;
		if (!obj->next) 
			cv.notify_all();
	}
};

#endif // FREE_LIST_ALLOCATOR_HH_INCLUDED
