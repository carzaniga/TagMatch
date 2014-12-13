#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <cassert>

#include "filter_set.hh"
#include "packet.hh"

struct node {
	node * zero_link;
	uint8_t pos;
	
	node() : zero_link(nullptr) {};
};

static node * root = nullptr;

static void clear_r(node * f) {
	if (!f)
		return;
	for(node * n = f; n->pos < filter_t::WIDTH; ++n) 
		clear_r(n->zero_link);
	delete[](f);
}

void filter_set::clear() {
	clear_r(root);
	root = nullptr;
}

void filter_set::add(const filter_t & f) {
	node ** hook = &root;

	unsigned int prefix_len = 0;
	unsigned int f_pos = f.next_bit(0); 

	while(*hook != nullptr) {
		node * n = *hook;
		while(n->pos == f_pos) {
			if (f_pos == filter_t::WIDTH)
				return;			// exact duplicate filter
			f_pos = f.next_bit(f_pos + 1);
			++prefix_len;
			++n;
		}
		assert((*hook)->pos < f_pos);
		hook = &(n->zero_link);
	}
	assert(prefix_len <= f.popcount());

	node * n = new node[f.popcount() + 1 - prefix_len];
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

bool filter_set::find(const filter_t & f) {
	node * n = root;
	unsigned int f_pos = f.next_bit(0);
	while(n != nullptr) {
		if (f_pos == n->pos) {
			if (f_pos == filter_t::WIDTH)
				return true;
			++n;
			f_pos = f.next_bit(f_pos + 1);
		} else if (f_pos < n->pos) {
			n = n->zero_link;
		} else
			return false;
	}
	return  false;
}

struct matching_point {
	node * n;
	unsigned int p;

	void assign(node * n_, unsigned int p_) {
		n = n_;
		p = p_;
	}
};

size_t filter_set::count_subsets_of(const filter_t & f) {
	size_t result = 0;

	matching_point stack[filter_t::WIDTH];
	unsigned int head = 0;


	unsigned int f_pos = f.next_bit(0);
	node * n = root;

	while (n != nullptr) {
		if(f_pos > n->pos) {
			n = n->zero_link;
			continue;
		} else if (f_pos == n->pos) {
			if (f_pos == filter_t::WIDTH) {
				++result;
			} else {
				f_pos = f.next_bit(f_pos + 1);
				if (n->zero_link != nullptr)
					stack[head++].assign(n->zero_link, f_pos);
				++n;
				continue;
			}
		} 
		if (head == 0)
			return result;
		--head;
		n = stack[head].n;
		f_pos = stack[head].p;
	}
	return result;
}

size_t filter_set::count_supersets_of(const filter_t & x) {
	return 0;
}
