#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <vector>
#include <cassert>

#include "filter_set.hh"
#include "packet.hh"

class node {
public:
	uint8_t pos;

private:
	bool have_sibling_table;
	union {
		node * next_sibling;
		node ** siblings;
	};

static const unsigned int MIN_TABLE_SIZE = 16;

public:
	node() : have_sibling_table(false), next_sibling(nullptr) {};

	const node * sibling_greater_or_equal(uint8_t p) const {
		if (have_sibling_table) {
			return siblings[p];
		} else {
			const node * n = this;
			while (n->pos < p) {
				n = n->next_sibling;
				if (n == nullptr)
					break;
			}
			return n;
		}
	}

	void consolidate();

	static void append_filter(node ** hook, const filter_t & f);
};

node * root = nullptr;

void filter_set::clear() {
	root = nullptr;
}

void node::consolidate() {
	if (have_sibling_table)
		return;

	unsigned int siblings_count = 0;
	for(const node * n = next_sibling; n != nullptr; n = n->next_sibling)
		++siblings_count;

	if (siblings_count < MIN_TABLE_SIZE)
		return;

	node ** table = new node*[filter_t::WIDTH + 1];

	node * n = this;
	for(uint8_t i = 0; i <= filter_t::WIDTH; ++i) {
		table[i] = n;
		if (n != nullptr && i == n->pos) {
			node * next = n->next_sibling;
			n->have_sibling_table = true;
			n->siblings = table;
			if (n->pos < filter_t::WIDTH)
				(n + 1)->consolidate();
			n = next;
		}
	}
}

void node::append_filter(node ** hook, const filter_t & f) {
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
		hook = &(n->next_sibling);
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

void filter_set::add(const filter_t & f) {
	node::append_filter(&root, f);
}

void filter_set::consolidate() {
	if (root != nullptr)
		root->consolidate();
}

bool filter_set::find(const filter_t & f) {
#if 0
	node * n = root;
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

	const node * stack[filter_t::WIDTH];
	unsigned int head = 0;

	unsigned int f_pos = f.next_bit(root->pos);
	const node * n = root->sibling_greater_or_equal(f_pos);
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
