#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <sstream>
#include <string>

#include "io_util.hh"
#include "fib.hh"

std::ostream & ti_vector::write_binary(std::ostream & output) const {
	uint32_t vector_size = size();
	io_util_write_binary(output, vector_size);
	for(const_iterator i = begin(); i != end(); ++i)
		i->write_binary(output);
	return output;
}

std::istream & ti_vector::read_binary(std::istream & input) {
	uint32_t vector_size;
	if (!io_util_read_binary(input, vector_size))
		return input;

	resize(vector_size);
	for(iterator i = begin(); i != end(); ++i)
		if (!i->read_binary(input))
			return input;

	return input;
}

std::ostream & fib_entry::write_ascii(std::ostream & output) const {
	output << "+ ";
	ti_pairs[0].write_ascii(output);
	output << " ";
	filter.write_ascii(output);
	for(unsigned int i = 1; i < ti_pairs.size(); ++i) {
		output << " ";
		ti_pairs[i].write_ascii(output);
	}
	output << std::endl;
	return output;
}

std::istream & fib_entry::read_ascii(std::istream & input) {
	std::string line;
	if (std::getline(input, line)) {
		std::istringstream input_line(line);
		std::string command;
		input_line >> command;

		if (command == "+") {
			tree_interface_pair_io ti;

			if (! ti.read_ascii(input_line))
				return input;

			if (! filter.read_ascii(input_line))
				return input;

			ti_pairs.clear();

			do {
				ti_pairs.push_back(ti);
			} while (ti.read_ascii(input_line));
		}
	}
	return input;
}

std::ostream & partition_prefix::write_ascii(std::ostream & output) const {
	output << "p " << partition << ' ';
	filter.write_ascii(output, length);
	output << ' ' << size << std::endl;
	return output;
}

std::istream & partition_prefix::read_ascii(std::istream & input) {
	std::string line;
	if (std::getline(input, line)) {
		std::istringstream input_line(line);
		std::string command;
		input_line >> command;
		if (command == "p") {
			if(input_line >> partition) {
				std::string filter_string;
				if (input_line >> filter_string) {
					length = filter_string.size();
					std::istringstream input_filter_string(filter_string);
					filter.read_ascii(input_filter_string);
					input_line >> size;
				}
			}
		}
	}
	return input;
}

std::ostream & partition_fib_entry::write_ascii(std::ostream & output) const {
	output << "f " << partition << ' ';
	filter.write_ascii(output);
	for(unsigned int i = 0; i < ti_pairs.size(); ++i) {
		output << " ";
		ti_pairs[i].write_ascii(output);
	}
	output << std::endl;
	return output;
}

std::istream & partition_fib_entry::read_ascii(std::istream & input) {
	std::string line;
	if (std::getline(input, line)) {
		std::istringstream input_line(line);
		std::string command;
		input_line >> command;

		if (command == "f") {
			if (input_line >> partition) {
				tree_interface_pair_io ti;

				if (! filter.read_ascii(input_line))
					return input;

				ti_pairs.clear();
				while(ti.read_ascii(input_line))
					ti_pairs.push_back(ti);
			}
		}
	}
	return input;
}
