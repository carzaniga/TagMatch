#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>

#include "filter.hh"
#include "routing.hh"

static inline std::ostream & operator << (std::ostream & os, const filter_t & f) {
	return f.write_ascii(os);
}

static std::vector<tree_interface_pair> ti_pairs;

struct filter_descr {
	filter_t filter;
	unsigned int ti_pairs_begin;
	unsigned int ti_pairs_end;

	filter_descr(const std::string & f, unsigned int b, unsigned int e)
		: filter(f), ti_pairs_begin(b), ti_pairs_end(e) {};

	filter_descr(const filter_descr & d)
		: filter(d.filter), ti_pairs_begin(d.ti_pairs_begin), ti_pairs_end(d.ti_pairs_end) {};

	filter_descr & operator = (const filter_descr & d) {
		filter = d.filter;
		ti_pairs_begin = d.ti_pairs_begin;
		ti_pairs_end = d.ti_pairs_end;
		return *this;
	}
};

static bool compare_filters_decreasing(const filter_descr & d1, const filter_descr & d2) {
	return d2.filter < d1.filter;
}

static std::vector<filter_descr> filters;

void read_filters_vector(std::istream & is) {
	bool sorted = true;
	std::string line;

	while(std::getline(is, line)) {
		std::istringstream line_s(line);
		
		std::string command;

		line_s >> command;

		if (command != "+")
			continue;

		unsigned int iface, tree;
		std::string filter;

		line_s >> tree >> iface >> filter;

		unsigned int begin = ti_pairs.size();

		do {
			ti_pairs.push_back(tree_interface_pair(tree, iface));
		} while (line_s >> tree >> iface);

		unsigned int end = ti_pairs.size();

		filters.push_back(filter_descr(filter, begin, end));
		if (sorted && filters.size() > 1) {
			sorted = (filters[filters.size() - 1].filter < filters[filters.size() - 2].filter);
		}
	}

	if (!sorted)
		std::sort(filters.begin(), filters.end(), compare_filters_decreasing);
}

static unsigned int MIN_K = 2;

filter_pos_t kth_most_significant_one_pos(const filter_t & f, unsigned int k) {
	filter_pos_t i = filter_t::WIDTH;
	do {
		--i;
		if (f[i]) 
			--k;
	} while (k > 0 && i > 0);
	return i;
}

void split_on_prefix(unsigned int max_size, std::ostream * prefix_os, std::ostream * filters_os) {
	std::ostringstream foss;
	std::vector<filter_descr>::const_iterator f = filters.begin();
	unsigned int pid = 0;

	while (f != filters.end()) {
		filter_pos_t prefix_pos = 0;
		std::vector<filter_descr>::const_iterator g = f + 1;
		std::vector<filter_descr>::const_iterator next_f = g; 

		filter_pos_t kth_msb_pos = kth_most_significant_one_pos(f->filter, MIN_K);
		
		while(g != filters.end() && (g - f) < max_size) {
			filter_pos_t msd = f->filter.leftmost_diff(g->filter);

			if (msd > prefix_pos) {
				if (kth_msb_pos <= msd) {
					next_f = g;
					prefix_pos += 1;
					break;
				}
				prefix_pos = msd;
				next_f = g;
			}
			++g;
		}
		if (g == filters.end()) {
			next_f = g;
			prefix_pos += 1;
		}

		foss.str("");
		foss << f->filter;
		if (prefix_os) {
			*prefix_os << "p " << pid 
					   << ' ' << foss.str().substr(0, (filter_t::WIDTH - prefix_pos))
					   << ' ' << (next_f - f) << std::endl;
		}
		if (filters_os) {
			for(std::vector<filter_descr>::const_iterator i = f; i != next_f; ++i) {
				*filters_os << "f " << pid << ' ' << i->filter;
				for(unsigned int j = i->ti_pairs_begin; j < i->ti_pairs_end; ++j)
					*filters_os << ' ' << ti_pairs[j].tree() << ' ' << ti_pairs[j].interface();
				*filters_os << std::endl;
			}
		}
		++pid;
		f = next_f;
	}
}

int main(int argc, const char * argv[]) {
	unsigned int max_size = 100;

	const char * prefixes_fname = 0;
	const char * filters_fname = 0;

	for(int i = 1; i < argc; ++i) {
		if (sscanf(argv[i],"m=%u", &max_size) || sscanf(argv[i],"N=%u", &max_size))
			continue;

		if (sscanf(argv[i],"k=%u", &MIN_K) || sscanf(argv[i],"K=%u", &MIN_K))
			continue;

		if (strncmp(argv[i],"p=",2)==0) {
			prefixes_fname = argv[i] + 2;
			continue;
		}

		if (strncmp(argv[i],"f=",2)==0) {
			filters_fname = argv[i] + 2;
			continue;
		}
		std::cerr << "usage: " << argv[0] << " [<params>...]\n"
			"\n  params: any combination of the following:\n"
			"     [m=<N>]         :: size limit for each partition (default=100)\n"
			"     [p=<filename>]  :: output for prefixes, '-' means stdout (default=OFF)\n"
			"     [f=<filename>]  :: output for filters, '-' means stdout (default=OFF)\n"
				  << std::endl;
		return 1;
	}

	std::ostream * prefixes_os = 0;
	std::ostream * filters_os = 0;

	std::ofstream prefixes_f;
	std::ofstream filters_f;

	if (filters_fname) {
		if (strcmp(filters_fname, "-")==0) {
			filters_os = &std::cout;
		} else {
			filters_f.open(filters_fname);
			if (!filters_f) {
				std::cerr << "error opening filters file "  << filters_fname << std::endl;
				return -1;
			}
			filters_os = &filters_f;
		}
	}

	if (prefixes_fname) {
		if (strcmp(prefixes_fname, "-")==0) {
			prefixes_os = &std::cout;
		} else {
			prefixes_f.open(prefixes_fname);
			if (!prefixes_f) {
				std::cerr << "error opening prefixes file "  << prefixes_fname << std::endl;
				return -1;
			}
			prefixes_os = &prefixes_f;
		}
	} 

	read_filters_vector(std::cin);
	split_on_prefix(max_size, prefixes_os, filters_os);

	return 0;
}
