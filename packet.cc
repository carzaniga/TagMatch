#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <sstream>
#include <string>

#include "packet.hh"

std::ostream & network_packet::write_binary(std::ostream & output) const {
	filter.write_binary(output);
	return output;
}

std::istream & network_packet::read_binary(std::istream & input) {
	filter.read_binary(input);
	return input;
}

std::ostream & network_packet::write_ascii(std::ostream & output) const {
	output.put('!');
	output.put(' ');
	filter.write_ascii(output);
	output.put('\n');
	return output;
}

std::istream & network_packet::read_ascii(std::istream & input) {
	std::string line;
	if(std::getline(input, line)) {
		std::istringstream line_s(line);
		std::string command;
		line_s >> command;
		if (command == "!") {
			filter.read_ascii(line_s);
		}
	}
	return input;
}
