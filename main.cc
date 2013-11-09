#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <string>

#include "predicate.hh"

class filter_printer : public filter_const_handler {
public:
	filter_printer(std::ostream & s): os(s) {};

	virtual bool handle_filter(const filter_t & filter, const predicate::node & n);
private:
	std::ostream & os;
};

bool filter_printer::handle_filter(const filter_t & filter, const predicate::node & n) {
	os << filter << std::endl;
	return false;
}

class match_printer : public match_handler {
public:
	match_printer(std::ostream & s): os(s) {};

	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
private:
	std::ostream & os;
};

bool match_printer::match(const filter_t & filter, tree_t tree, interface_t ifx) {
	os << "-> " << ifx << ' ' << filter << std::endl;
	return false;
}

class match_counter : public match_handler {
public:
	match_counter(): count(0) {};

	virtual bool match(const filter_t & filter, tree_t tree, interface_t ifx);
	unsigned long get_match_count() const {
		return count;
	}
private:
	unsigned long count;
};

bool match_counter::match(const filter_t & filter, tree_t tree, interface_t ifx) {
	++count;
	return false;
}

inline bool wheel_of_death(unsigned long counter, unsigned int mask_bits) {
	static const char * wheel = "-\\|/";

	if (counter & ((1UL << mask_bits) - 1))
		return false;

	std::cout << wheel[(counter >> mask_bits) & 3];
	return true;
}

int main(int argc, char *argv[]) {

	std::string command;
	std::string tree;
	std::string interface;
	std::string filter_string;
	
	predicate P;

	filter_printer filter_output(std::cout);
	match_printer match_output(std::cout);
	match_counter match_count;

	unsigned int count = 0;
	unsigned int query_count = 0;


	while(std::cin >> command >> tree >> interface >> filter_string) {
		if (command == "+") {
			filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
			P.add(filter,t,i);
			++count;
			if (wheel_of_death(count, 10))
				std::cout << " N=" << count << "  Unique=" << P.size() << "\r";

		} else if (command == "+q") {
			filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
			P.add(filter,t,i);
			++count;
		} else if (command == "?") {
			filter_t filter(filter_string);
			std::cout << "==== " << filter << std::endl;
			tree_t t = atoi(tree.c_str());
			P.match(filter, t, match_output);
		} else if (command == "!") {
			filter_t filter(filter_string);
			tree_t t = atoi(tree.c_str());
			P.match(filter, t, match_count);
			if (query_count==0)
				std::cout << std::endl;
			++query_count;
			if (wheel_of_death(query_count, 7))
				std::cout << " Q=" << query_count 
						  << "  Match=" << match_count.get_match_count() << " \r";
		} else if (command == "sup") {
			filter_t filter(filter_string);
			P.find_supersets_of(filter, filter_output);
		} else if (command == "sub") {
			filter_t filter(filter_string);
			P.find_subsets_of(filter, filter_output);
		} else {
			std::cerr << "unknown command: " << command << std::endl;
		}
	}
	std::cout << std::endl << "Final statistics:" << std::endl
			  << "N=" << count << "  Unique=" << P.size()  << std::endl
			  << "Q=" << query_count << "  Match=" << match_count.get_match_count() << std::endl;

	return 0;
}