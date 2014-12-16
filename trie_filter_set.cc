#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define POINTERS_ARE_INDEXES 1

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

	pointer(uint32_t i) : ptr(i) {};

public:
	pointer() {};
	pointer(const pointer & x) : ptr(x.ptr) {};
	pointer(T * p) {
		if(!p) {
			ptr = NULL_PTR;
		} else {
			ptr = p - &(memory[0]);
		}
	};

	static void deallocate_all() {
		memory.clear();
	}

	static pointer allocate(size_t n) {
		uint32_t i = memory.size();
		memory.resize(i + n);
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

	bool operator == (T * p) const {
		assert(p == nullptr);
		return ptr == NULL_PTR;
	}

	bool operator != (T * p) const {
		assert(p == nullptr);
		return ptr != NULL_PTR;
	}

	size_t operator - (const pointer & x) const {
		return x - x.ptr;
	}

	T & operator * () const {
		return memory[ptr];
	}

	T * operator->() const {
		return &(memory[ptr]);
	}

	T & operator[] (size_t i) const {
		return memory[ptr + i];
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
		return ptr != NULL_PTR;
	}
};

template<typename T>
typename std::vector<T> pointer<T>::memory;

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
		if (current_chunk_size + n > CHUNK_SIZE || (!current_chunk)) {
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

static const unsigned int MIN_TABLE_SIZE = 8;

public:
	node(const node & x) : pos(x.pos), have_sibling_table(x.have_sibling_table) {
		if (have_sibling_table) {
			next_sibling = x.next_sibling;
		} else {
			siblings = x.siblings;
		}
	}

	node() : have_sibling_table(false), next_sibling(nullptr) {};

	pointer<node> sibling_greater_or_equal(uint8_t p) {
		if (have_sibling_table) {
			return siblings[p];
		} else {
			if (pos >= p)
				return pointer<node>(this);

			pointer<node> n = next_sibling;
			while((n) && n->pos < p)
				n = n->next_sibling;

			return n;
		}
	}

	static void consolidate(pointer<node> n);

	static void append_filter(pointer<node> n, const filter_t & f);
};

pointer<node> root = nullptr;

void filter_set::clear() {
	pointer<node>::deallocate_all();
	pointer< pointer<node> >::deallocate_all();
	root = nullptr;
}

void node::consolidate(pointer<node> n) {
	if (n->have_sibling_table)
		return;

	unsigned int siblings_count = 0;
	for(pointer<node> ns = n->next_sibling; (ns); ns = ns->next_sibling)
		++siblings_count;

	if (siblings_count < MIN_TABLE_SIZE)
		return;

	pointer< pointer<node> > table = pointer< pointer<node> >::allocate(filter_t::WIDTH + 1);

	for(uint8_t i = 0; i <= filter_t::WIDTH; ++i) {
		table[i] = n;
		if ((n) && i == n->pos) {
			pointer<node> next = n->next_sibling;
			n->have_sibling_table = true;
			n->siblings = table;
			if (n->pos < filter_t::WIDTH) {
				pointer<node> np = n;
				consolidate(++np);
			}
			n = next;
		}
	}
}

pointer<node> new_chain(const filter_t & f, unsigned int pos, unsigned int len) {
	pointer<node> res = pointer<node>::allocate(len);
	pointer<node> n = res;
	for(;;) {
		n->pos = pos;
		if (pos < filter_t::WIDTH) {
			++n;
			pos = f.next_bit(pos + 1);
		} else
			return res;
	}
}

void node::append_filter(pointer<node> n, const filter_t & f) {
	unsigned int len = f.popcount() + 1;
	unsigned int f_pos = f.next_bit(0); 

	assert((n) && n->pos <= f_pos);

	for(;;) {
		while(n->pos == f_pos) {
			if (f_pos == filter_t::WIDTH)
				return;			// exact duplicate filter
			f_pos = f.next_bit(f_pos + 1);
			--len;
			++n;
		}
		assert(len > 0);
		assert(n->pos < f_pos);
		if (!n->next_sibling) {
			n->next_sibling = new_chain(f, f_pos, len);
			return;
		} else {
			n = n->next_sibling;
		}
	}
}

void filter_set::add(const filter_t & f) {
	if (root) {
		node::append_filter(root, f);
	} else {
		root = new_chain(f, f.next_bit(0), f.popcount() + 1);
	}
}

void filter_set::consolidate() {
	if (root)
		node::consolidate(root);
}

bool filter_set::find(const filter_t & f) {
#if 0
	pointer<node> n = root;
	unsigned int f_pos = f.next_bit(0);
	while(n) {
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
	if (!n)
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
			if (n)
				continue;
		}
		// pop node from stack
		do {
			if (head == 0)
				return result;
			n = stack[--head];
			f_pos = f.next_bit(n->pos + 1);
			n = n->sibling_greater_or_equal(f_pos);
		} while (!n);
	}
}

size_t filter_set::count_supersets_of(const filter_t & x) {
	return 0;
}
