#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <string>
#include <chrono>
#include <math.h> 

#include "router.hh"



#define WITH_INFO_OUTPUT 0
#define TIME_DISTRUBUTION 1

using namespace std::chrono;


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

    std::string wkld_name;
	
#if TIME_DISTRUBUTION
    //argv[1] contains the number of expected filters
    if(argc<3){
        std::cout << "expected number of filters missing (or WKLD_NAME)" << std::endl;
        exit(0);
    }
	router R(atoi(argv[1]));
    wkld_name = argv[2];
#else
    if(argc<2){
        std::cout << "expected number of filters missing" << std::endl;
        exit(0);
    }
	router R(atoi(argv[1]));
#endif
    R.start_threads();


    unsigned int update_count =0;
	unsigned int count = 0;
    unsigned int add = 0;
    unsigned int rm = 0;

#if WITH_INFO_OUTPUT
    unsigned int tot_out=0;
#endif

    
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point stop;
    
    vector<unsigned long> time_array;
    microseconds total_time;

	
	while(std::cin >> command >> tree >> interface >> filter_string) {
		if (command == "+") {
			filter_t filter(filter_string);
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
            R.add_filter_without_check(filter,t,i);
			++count;
#if WITH_INFO_OUTPUT
            if((count % 1000)==0)
                std::cout << " N=" << count << "\r";
#endif
		}else if (command =="#"){
             //this does not work fine          
        }else if (command == "sd"){ //start delta tree ifx random_val
            predicate_delta pd;
			interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
            std::cin >> command >> tree >> interface >> filter_string;
            while(command!="ed"){ //end delta random_val random_val random_val
                filter_t filter(filter_string);
                if(command=="d+"){ 
                    add++;
                    pd.additions.push_back(filter);
                }else if(command=="d-"){
                    rm++;
                    pd.removals.push_back(filter);
                }else if (command =="#"){
                   
                }
                std::cin >> command >> tree >> interface >> filter_string;
            }
            map<interface_t,predicate_delta> out;

            start = std::chrono::high_resolution_clock::now();
		
            R.apply_delta(out, pd, i, t);

            stop = high_resolution_clock::now();

#if WITH_INFO_OUTPUT
            
            tot_out+=out.size();
            //cout << "out size:" << out.size() <<endl;
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

            microseconds ms = duration_cast<microseconds>(stop - start);
            time_array.push_back(ms.count());
            total_time+=ms;

            update_count++;

#if WITH_INFO_OUTPUT	
		    std::cout << " N=" << update_count << " add=" << add << " rm=" << rm
            << " Tu (us)=" << (total_time.count()/update_count) 
            << "\r";
#endif

        }else if(command == "+ti"){ //command tree ifx random_val
            interface_t i = atoi(interface.c_str());
			tree_t t = atoi(tree.c_str());
            R.add_ifx_to_tree(t,i);
        }else {
#if WITH_INFO_OUTPUT
			std::cerr << "unknown command: " << command << std::endl;
#endif
		}
	}
#if WITH_INFO_OUTPUT

	std::cout << std::endl << "Final statistics:" << std::endl;
              std::cout << "Filters=" << count << std::endl;  
              std::cout << "Updates=" << update_count << " add=" << add << " rm=" << rm << std::endl;
			  std::cout << "Tu (us)=" << (total_time.count() / update_count) << std::endl;

    std::cout << "Updates Generated=" << tot_out << std::endl;
#endif

#if TIME_DISTRUBUTION        
    for (unsigned int i =0; i<time_array.size(); i++){
        std::cout << time_array[i] << std::endl;
    }
#endif

    R.stop_threads();

	return 0;
}
