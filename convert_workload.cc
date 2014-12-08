#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>

#include "fib.hh"
#include "packet.hh"

int convert_filters(std::istream & input, std::ostream & output) {
	fib_entry e;
	while(e.read_ascii(input))
		e.write_binary(output);

	return 0;
}

int convert_filters_ascii(std::istream & input, std::ostream & output) {
	fib_entry e;
	while(e.read_binary(input))
		e.write_ascii(output);

	return 0;
}

int convert_queries(std::istream & input, std::ostream & output) {
	network_packet packet;

	while(packet.read_ascii(input))
		packet.write_binary(output);

	return 0;
}

int convert_queries_ascii(std::istream & input, std::ostream & output) {
	network_packet packet;

	while(packet.read_binary(input))
		packet.write_ascii(output);

	return 0;
}

int convert_prefixes(std::istream & input, std::ostream & output) {
	partition_prefix p;
	while(p.read_ascii(input))
		p.write_binary(output);
	return 0;
}

int convert_prefixes_ascii(std::istream & input, std::ostream & output) {
	partition_prefix p;
	while(p.read_binary(input))
		p.write_ascii(output);
	return 0;
}

int convert_partitions(std::istream & input, std::ostream & output) {
	return 0;
}

int convert_partitions_ascii(std::istream & input, std::ostream & output) {
	return 0;
}

enum conversion_type {
	QUERIES = 0,
	FILTERS = 1,
	PREFIXES = 2,
	PARTITIONS = 3
};

class conversion {
private:
	bool to_ascii;
	conversion_type type;

public:
	conversion(): to_ascii(false), type(FILTERS) {};

	void set_ascii_output() { 
		to_ascii = true;
	}

	void set_conversion(conversion_type t) { 
		type = t;
	}

	int run(std::istream & input, std::ostream & output) {
		if (to_ascii) {
			switch(type) {
			case QUERIES: return convert_queries_ascii(input, output);
			case FILTERS: return convert_filters_ascii(input, output);
			case PREFIXES: return convert_prefixes_ascii(input, output);
			case PARTITIONS: return convert_partitions_ascii(input, output);
			}
			return 1;
		} else {
			switch(type) {
			case QUERIES: return convert_queries(input, output);
			case FILTERS: return convert_filters(input, output);
			case PREFIXES: return convert_prefixes(input, output);
			case PARTITIONS: return convert_partitions(input, output);
			}
			return 1;
		}
	}
};

static void print_usage(const char * progname) {
	std::cout << "usage: " << progname 
			  << " [in=<filename>] [out=<filename>] [options...]"
			  << std::endl
			  << "options:" << std::endl
			  << "\t-a\t: converts from binary to ASCII (default is ASCII to binary)" << std::endl
			  << "\t-f\t: converts filters" << std::endl
			  << "\t-q\t: converts queries" << std::endl
			  << "\t-p\t: converts prefixes" << std::endl;
}

int main(int argc, const char * argv[]) {
	conversion conv;

	const char * input_fname = nullptr;
	const char * output_fname = nullptr;

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"in=",3)==0) {
			input_fname = argv[i] + 3;
		} else 
		if (strncmp(argv[i],"out=",4)==0) {
			output_fname = argv[i] + 4;
		} else 
		if (strcmp(argv[i],"-a")==0) {
			conv.set_ascii_output();
		} else 
		if (strcmp(argv[i],"-f")==0) {
			conv.set_conversion(FILTERS);
		} else 
		if (strcmp(argv[i],"-q")==0) {
			conv.set_conversion(QUERIES);
		} else 
		if (strcmp(argv[i],"-p")==0) {
			conv.set_conversion(PREFIXES);
		} else {
			print_usage(argv[0]);
			return 1;
		}
	}

	int res;

	if (input_fname != nullptr) {
		std::ifstream input_file(input_fname);
		if (!input_file) {
			std::cerr << "could not open input file " << input_fname << std::endl;
			return 1;
		}
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = conv.run(input_file, output_file);
			output_file.close();
		} else {
			res = conv.run(input_file, std::cout);
		}
		input_file.close();
	} else {
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = conv.run(std::cin, output_file);
			output_file.close();
		} else {
			res = conv.run(std::cin, std::cout);
		}
	}

	return res;
}
