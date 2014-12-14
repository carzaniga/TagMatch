#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

// #define POINTERS_ARE_INDEXES 1

#include <cstddef>
#include <cassert>
#if POINTERS_ARE_INDEXES
#include <vector>
#endif

#include "filter_set.hh"
#include "packet.hh"

#if POINTERS_ARE_INDEXES
template<typename T>
class pointer {
private:
	uint32_t ptr;

	static const uint32_t NULL_PTR = 0xffffffff;

	static std::vector<T> memory;

public:
	pointer() {};
	pointer(const pointer & x) : ptr(x.ptr) {};
	pointer(T * p) : ptr(p) {
		if (p == nullptr) 
			ptr = NULL_PTR;
		else
			ptr = p - &(memory[0]);
	};

	static void deallocate_all() {
		memory.clear();
	}

	static pointer allocate(size_t n) {
		uint32_t i = memory.size();
		memory.resize(n);
		return pointer(i);
	}

	pointer & operator = (const pointer & x) {
		ptr = x.ptr;
		return *this;
	}

	bool operator == (const pointer & x) const {
		return ptr == x.ptr;
	}

	bool operator != (const pointer & x) const {
		return ptr != x.ptr;
	}

	size_t operator - (const pointer & x) const {
		return x - x.ptr;
	}

	T & operator * () const {
		return memory[ptr];
	}

	T * operator->() const {
		return &(memory[ptr];
	}

	T & operator[] (size_t i) const {
		return memory[ptr + i];
	}

	pointer operator+ (size_t i) const {
		return pointer{ptr + i};
	}

	pointer & operator++ () {
	    ++ptr;
		return *this;
	}

	pointer operator++ (int) {
		return pointer(ptr++);
	}

	operator bool () {
		return ptr != NULL_PTR;
	}
};

template<typename T>
typename std::vector<T>:: pointer<T>::memory;

#else
template<typename T>
class pointer {
private:
	T * ptr;
	static const size_t CHUNK_SIZE = 0x10000; // 64K objects

	struct chunk {
		chunk * next;
		T objects[CHUNK_SIZE];
	};

	static size_t current_chunk_size;
	static chunk * current_chunk;

public:
	pointer() {};
	pointer(const pointer & x) : ptr(x.ptr) {};
	pointer(T * p) : ptr(p) {};

	static void deallocate_all() {
		while(current_chunk != nullptr) {
			chunk * next = current_chunk->next;
			delete(current_chunk);
			current_chunk = next;
		}
		current_chunk_size = 0;
	}

	static pointer allocate(size_t n) {
		if (current_chunk_size + n > CHUNK_SIZE || current_chunk == nullptr) {
			chunk * new_chunk = new chunk();
			new_chunk->next = current_chunk;
			current_chunk = new_chunk;
			current_chunk_size = 0;
		}
		T * p = current_chunk->objects + current_chunk_size;
		current_chunk_size += n;
		return pointer(p);
	}

	pointer & operator = (const pointer & x) {
		ptr = x.ptr;
		return *this;
	}

	pointer & operator = (T * p) {
		ptr = p;
		return *this;
	}

	bool operator == (const pointer & x) const {
		return ptr == x.ptr;
	}

	bool operator != (const pointer & x) const {
		return ptr != x.ptr;
	}

	size_t operator - (const pointer & x) const {
		return index - x.index;
	}

	T & operator * () const {
		return *ptr;
	}

	T * operator->() const {
		return ptr;
	}

	T & operator[] (size_t i) const {
		return *(ptr + i);
	}

	pointer operator+ (size_t i) const {
		return pointer(ptr + i);
	}

	pointer & operator++ () {
		++ptr;
		return *this;
	}

	pointer operator++ (int) {
		return pointer(ptr++);
	}

	operator bool () {
		return ptr != nullptr;
	}
};

template<typename T>
size_t pointer<T>::current_chunk_size = 0;

template<typename T>
typename pointer<T>::chunk * pointer<T>::current_chunk = nullptr;

#endif

class node {
public:
	uint8_t pos;

private:
	bool have_sibling_table;
	union {
		pointer<node> next_sibling;
		pointer< pointer<node> > siblings;
	};

static const unsigned int MIN_TABLE_SIZE = 4;

public:
	node() : have_sibling_table(false), next_sibling(nullptr) {};

	pointer<node> sibling_greater_or_equal(uint8_t p) {
		if (have_sibling_table) {
			return siblings[p];
		} else {
			pointer<node> n = this;
			while (n->pos < p) {
				n = n->next_sibling;
				if (n == nullptr)
					break;
			}
			return n;
		}
	}

	void consolidate();

	static void append_filter(pointer<node>* hook, const filter_t & f);
};

pointer<node> root = nullptr;

void filter_set::clear() {
	pointer<node>::deallocate_all();
	pointer< pointer<node> >::deallocate_all();
	root = nullptr;
}

void node::consolidate() {
	if (have_sibling_table)
		return;

	unsigned int siblings_count = 0;
	for(pointer<node> n = next_sibling; n != nullptr; n = n->next_sibling)
		++siblings_count;

	if (siblings_count < MIN_TABLE_SIZE)
		return;

	pointer< pointer<node> > table = pointer< pointer<node> >::allocate(filter_t::WIDTH + 1);

	pointer<node> n = this;
	for(uint8_t i = 0; i <= filter_t::WIDTH; ++i) {
		table[i] = n;
		if (n != nullptr && i == n->pos) {
			pointer<node> next = n->next_sibling;
			n->have_sibling_table = true;
			n->siblings = table;
			if (n->pos < filter_t::WIDTH)
				n[1].consolidate();
			n = next;
		}
	}
}

void node::append_filter(pointer<node>* hook, const filter_t & f) {
	unsigned int prefix_len = 0;
	unsigned int f_pos = f.next_bit(0); 

	while(*hook != nullptr) {
		pointer<node> n = *hook;
		while(n->pos == f_pos) {
			if (f_pos == filter_t::WIDTH)
				return;			// exact duplicate filter
			f_pos = f.next_bit(f_pos + 1);
			++prefix_len;
			++n;
		}
		assert((*hook)->pos < f_pos);
		hook = &(n->next_sibling);
	}
	assert(prefix_len <= f.popcount());

	pointer<node> n = pointer<node>::allocate(f.popcount() + 1 - prefix_len);
	*hook = n;

	for(;;) {
		n->pos = f_pos;
		if (f_pos < filter_t::WIDTH) {
			++n;
			f_pos = f.next_bit(f_pos + 1);
		} else
			break;
	}
}

void filter_set::add(const filter_t & f) {
	node::append_filter(&root, f);
}

void filter_set::consolidate() {
	if (root != nullptr)
		root->consolidate();
}

bool filter_set::find(const filter_t & f) {
#if 0
	pointer<node> n = root;
	unsigned int f_pos = f.next_bit(0);
	while(n != nullptr) {
		if (f_pos == n->pos) {
			if (f_pos == filter_t::WIDTH)
				return true;
			++n;
			f_pos = f.next_bit(f_pos + 1);
		} else if (f_pos < n->pos) {
			n = n->next_sibling;
		} else
			return false;
	}
#endif
	return  false;
}

size_t filter_set::count_subsets_of(const filter_t & f) {
	if (!root)
		return 0;

	size_t result = 0;

	pointer<node> stack[filter_t::WIDTH];
	unsigned int head = 0;

	unsigned int f_pos = f.next_bit(root->pos);
	pointer<node> n = root->sibling_greater_or_equal(f_pos);
	if (n == nullptr)
		return 0;

	for (;;) { // assert(f_pos <= n->pos);
		if (f_pos == n->pos) {
			if (f_pos == filter_t::WIDTH) {
				++result;
			} else {
				stack[head++] = n;
				++n;
				f_pos = f.next_bit(n->pos);
				continue;
			}
		} else { // assert(f_pos < n->pos);
			f_pos = f.next_bit(n->pos);
			n = n->sibling_greater_or_equal(f_pos);
			if (n != nullptr)
				continue;
		}
		// pop node from stack
		do {
			if (head == 0)
				return result;
			n = stack[--head];
			f_pos = f.next_bit(n->pos + 1);
			n = n->sibling_greater_or_equal(f_pos);
		} while (n == nullptr);
	}
}

size_t filter_set::count_supersets_of(const filter_t & x) {
	return 0;
}
