// EMACS indentation settings (please, do not delete)
//
// Local Variables:
// c-file-style: "linux"
// indent-tabs-mode: t
// tab-width: 8
// End:
//
#ifndef PREDICATE_H_INCLUDED
#define PREDICATE_H_INCLUDED

#include <iostream>
#include <set>

#include "filter.h"

using namespace std;

class TreeIffPair;

class node;
class end_node;
class predicate {
private:
	node * root;

public:
predicate(): root(0) {};
	void add_filter(const filter & f, unsigned char tree, unsigned char iff,const string & bitstring);
	void findMatch(const filter & f,int tree, const string & bitstring);// const;
	bool contains_subset(const filter & f) const;
	unsigned long count_nodes() const;
	unsigned long count_interfaces() const;

	friend ostream & operator << (ostream &, const predicate &);
	ostream & print(ostream &) const;
};

inline ostream & operator << (ostream & os, const predicate & p) {
	return p.print(os);
}

#endif
