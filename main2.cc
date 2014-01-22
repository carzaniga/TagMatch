#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <string>

#include "router.hh"
#include "timing.hh"

#define PRINT 0

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
	
    //argv[1] contains the number of expected filters
    if(argc<2){
        std::cout << "expected number of filters missing" << std::endl;
        exit(0);
    }
	router R(atoi(argv[1]));

	filter_printer filter_output(std::cout);
	match_printer match_output(std::cout);
	match_counter match_count;

	unsigned int count = 0;
	unsigned int query_count = 0;
    unsigned int update_count = 0;
    unsigned int match_c = 0;
    unsigned int add = 0;
    unsigned int rm = 0;

    
#ifdef WITH_TIMERS
	unsigned long long prev_nsec = 0;
    unsigned long long prev_match =0;
	Timer add_timer, update_timer, delete_timer, match_timer;
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
#if PRINT
			if (wheel_of_death(count, 12))
                std::cout << " N=" << count << "\r";
#endif
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
		} else if (command == "!") {
            filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
#ifdef WITH_TIMERS
			match_timer.start();
#endif
			R.match (filter,t,i);
#ifdef WITH_TIMERS
			match_timer.stop();
#endif
            match_c++;
#if PRINT
			if (wheel_of_death(match_c, 7))
               cout <<"Matches "<< match_c << " Tm (us)=" << ((match_timer.read_nanoseconds()/1000 - prev_match) >> 7) << "\r";
            prev_match=match_timer.read_nanoseconds()/1000;
#endif
            
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
		}else if (command =="#"){
            std::cin.ignore(5000,'\n');            
        }else if (command == "sd"){ //start delta tree ifx random_val
            predicate_delta pd(atoi(interface.c_str()), atoi(tree.c_str()));
            std::cin >> command >> tree >> interface >> filter_string;
            while(command!="ed"){ //end delta random_val random_val random_val
                filter_t filter(filter_string);
                if(command=="d+"){ 
                    add++;
                    pd.additions.insert(filter);
                }else if(command=="d-"){
                    rm++;
                    pd.removals.insert(filter);
                }else if (command =="#")
                    std::cin.ignore(5000,'\n'); 
                std::cin >> command >> tree >> interface >> filter_string;
            }
            vector<predicate_delta> out;
#ifdef WITH_TIMERS
			update_timer.start();
#endif
            R.apply_delta(out,pd);
#if 1
            cout << "out size:" << out.size() <<endl;
            /*for(vector<predicate_delta>::iterator it=out.begin(); it!=out.end(); it++){
                cout << "ifx:" << it->ifx << " tree:" << it->tree << " add:" << it->additions.size() << " rm:" 
                << it->removals.size() <<endl; 
                cout << "additions" << endl;
                for(set<filter_t>::iterator ii = it->additions.begin(); ii!=it->additions.end(); ii++)
                    cout << ii->print(cout) << endl;
                cout << "removals" << endl;
                for(set<filter_t>::iterator ii = it->removals.begin(); ii!=it->removals.end(); ii++)
                    cout << ii->print(cout) << endl;

            }*/
#endif

#ifdef WITH_TIMERS
			update_timer.stop();
#endif
        ++update_count;
#if PRINT
		if (wheel_of_death(update_count, 7)){
			//std::cout << " N=" << count << "  Unique=" << P.size() << "\r";
            std::cout << " N=" << update_count << " add=" << add << " rm=" << rm
#ifdef WITH_TIMERS
            << " Tu (us)=" << ((update_timer.read_nanoseconds()/1000 - prev_nsec) >> 7)
#endif
            << "\r";

#ifdef WITH_TIMERS
            prev_nsec = update_timer.read_nanoseconds()/1000;
#endif
        }
#endif

        }else if(command == "+ti"){ //command tree ifx random_val
            interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
            R.add_ifx_to_tree(t,i);
        }else {
			std::cerr << "unknown command: " << command << std::endl;
		}
	}
	std::cout << std::endl << "Final statistics:" << std::endl
			  //<< "N=" << count << "  Unique=" << P.size() << std::endl
              << "N=" << count << std::endl  
			  << "Q=" << query_count << "  Match=" << match_count.get_match_count() << std::endl
              << "U=" << update_count << " add=" << add << " rm=" << rm << std::endl;
#ifdef WITH_TIMERS
	std::cout << "Ta (us)=" << (add_timer.read_microseconds() / count) << std::endl 
			  << "Tu (us)=" << (update_timer.read_microseconds() / update_count) << std::endl;
              std::cout<< "Tm (us)=" << (match_timer.read_microseconds() / match_c) << std::endl;
#endif

	return 0;
}
