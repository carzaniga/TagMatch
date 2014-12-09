#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>

#include "fib.hh"
#include "packet.hh"

static int convert_filters(std::istream & input, std::ostream & output, bool binary_input) {
	fib_entry e;
	if (binary_input) {
		while(e.read_binary(input))
			e.write_ascii(output);
	} else {
		while(e.read_ascii(input))
			e.write_binary(output);
	}
	return 0;
}

static int convert_queries(std::istream & input, std::ostream & output, bool binary_input) {
	network_packet packet;

	if (binary_input) {
		while(packet.read_ascii(input))
			packet.write_binary(output);
	} else {
		while(packet.read_ascii(input))
			packet.write_binary(output);
	}
	return 0;
}

static int convert_prefixes(std::istream & input, std::ostream & output, bool binary_input) {
	partition_prefix p;
	if (binary_input) {
		while(p.read_binary(input))
			p.write_ascii(output);
	} else {
		while(p.read_ascii(input))
			p.write_binary(output);
	}
	return 0;
}

static int convert_partition_filters(std::istream & input, std::ostream & output, bool binary_input) {
	partition_fib_entry e;
	if (binary_input) {
		while(e.read_binary(input))
			e.write_ascii(output);
	} else {
		while(e.read_ascii(input))
			e.write_binary(output);
	}
	return 0;
}

static void print_usage(const char * progname) {
	std::cout << "usage: " << progname 
			  << " [in=<filename>] [out=<filename>] [options...]"
			  << std::endl
			  << "options:" << std::endl
			  << "\t-a\t: converts from binary to ASCII (default is ASCII to binary)" << std::endl
			  << "\t-F\t: converts global filters (default)" << std::endl
			  << "\t-q\t: converts queries" << std::endl
			  << "\t-p\t: converts prefixes" << std::endl
			  << "\t-f\t: converts partitioned filters" << std::endl;
}

int main(int argc, const char * argv[]) {
	bool binary_input = false;

	const char * input_fname = nullptr;
	const char * output_fname = nullptr;

	int (*conversion)(std::istream &, std::ostream &, bool) = convert_filters;

	for(int i = 1; i < argc; ++i) {
		if (strncmp(argv[i],"in=",3)==0) {
			input_fname = argv[i] + 3;
		} else 
		if (strncmp(argv[i],"out=",4)==0) {
			output_fname = argv[i] + 4;
		} else 
		if (strcmp(argv[i],"-a")==0) {
			binary_input = true;
		} else 
		if (strcmp(argv[i],"-f")==0) {
			conversion = convert_partition_filters;
		} else 
		if (strcmp(argv[i],"-F")==0) {
			conversion = convert_filters;
		} else 
		if (strcmp(argv[i],"-q")==0) {
			conversion = convert_queries;
		} else 
		if (strcmp(argv[i],"-p")==0) {
			conversion = convert_prefixes;
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
			res = conversion(input_file, output_file, binary_input);
			output_file.close();
		} else {
			res = conversion(input_file, std::cout, binary_input);
		}
		input_file.close();
	} else {
		if (output_fname != nullptr) {
			std::ofstream output_file(output_fname);
			if (!output_file) {
				std::cerr << "could not open output file " << output_fname << std::endl;
				return 1;
			}
			res = conversion(std::cin, output_file, binary_input);
			output_file.close();
		} else {
			res = conversion(std::cin, std::cout, binary_input);
		}
	}

	return res;
}
