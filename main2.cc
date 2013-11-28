#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <string>

#include "router.hh"
#include "timing.hh"

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
	
	router R;

	filter_printer filter_output(std::cout);
	match_printer match_output(std::cout);
	match_counter match_count;

	unsigned int count = 0;
	unsigned int query_count = 0;
    unsigned int update_count = 0;

    
#ifdef WITH_TIMERS
	//unsigned long long prev_nsec = 0;
	Timer add_timer, update_timer, delete_timer;
#endif
	
	while(std::cin >> command >> tree >> interface >> filter_string) {
		if (command == "+") {
			filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
#ifdef WITH_TIMERS
			add_timer.start();
#endif
			R.add_filter_without_check (filter,t,i);
#ifdef WITH_TIMERS
			add_timer.stop();
#endif
			++count;
			if (wheel_of_death(count, 12))
				//std::cout << " N=" << count << "  Unique=" << P.size() << "\r";
                std::cout << " N=" << count << "\r";

		} else if (command == "+q") {
			filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
#ifdef WITH_TIMERS
			add_timer.start();
#endif
			R.add_filter_without_check (filter,t,i);
#ifdef WITH_TIMERS
			add_timer.stop();
#endif
			++count;
		} else if (command =="-"){
			filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
#ifdef WITH_TIMERS
			delete_timer.start();
#endif
			R.remove_filter_without_check (filter,t,i);
#ifdef WITH_TIMERS
			delete_timer.stop();
#endif
		} else if (command =="+u"){
            filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
#ifdef WITH_TIMERS
			update_timer.start();
#endif
			R.add_filter(filter,t,i);
#ifdef WITH_TIMERS
			update_timer.stop();
#endif
			++update_count;
			if (wheel_of_death(count, 12))
				//std::cout << " N=" << count << "  Unique=" << P.size() << "\r";
                std::cout << " N=" << update_count << "\r";

        } else if (command =="-u"){
            filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
            set<predicate_delta> out;
#ifdef WITH_TIMERS
			update_timer.start();
#endif
			R.remove_filter(out,filter,t,i);
#ifdef WITH_TIMERS
			update_timer.stop();
#endif
			++update_count;
			if (wheel_of_death(count, 12))
				//std::cout << " N=" << count << "  Unique=" << P.size() << "\r";
                std::cout << " N=" << update_count << "\r";
        }else if (command =="#"){
            std::cin.ignore(500,'\n');            
        } else {
			std::cerr << "unknown command: " << command << std::endl;
		}
	}
	std::cout << std::endl << "Final statistics:" << std::endl
			  //<< "N=" << count << "  Unique=" << P.size() << std::endl
              << "N=" << count << std::endl  
			  << "Q=" << query_count << "  Match=" << match_count.get_match_count() << std::endl;
#ifdef WITH_TIMERS
	std::cout << "Ta (us)=" << (add_timer.read_microseconds() / count) << std::endl 
			  << "Tu+ (us)=" << (update_timer.read_microseconds() / update_count) << std::endl;
#endif

	return 0;
}
