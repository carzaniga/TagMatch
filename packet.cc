#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <sstream>
#include <string>

#include "packet.hh"

std::ostream & network_packet::write_binary(std::ostream & output) const {
	filter.write_binary(output);
	tip_write_binary(ti_pair, output);
	return output;
}

std::istream & network_packet::read_binary(std::istream & input) {
	filter.read_binary(input);
	tip_read_binary(ti_pair, input);
	return input;
}

std::ostream & network_packet::write_ascii(std::ostream & output) const {
	output.put('!');
	output.put(' ');
	tip_write_ascii(ti_pair, output);
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
			if (tip_read_ascii(ti_pair, line_s))
				filter.read_ascii(line_s);
		}
	}
	return input;
}

std::ostream & tip_write_binary(const tree_interface_pair & tip, std::ostream & output) {
	return io_util_write_binary(output, tip.get_uint16_value());
}

std::istream & tip_read_binary(tree_interface_pair & tip, std::istream & input) {
	uint16_t v;
	if (io_util_read_binary(input, v))
		tip.assign_uint16_value(v);
	return input;
}

std::ostream & tip_write_ascii(const tree_interface_pair & tip, std::ostream & output) {
	return output << tip.tree() << ' ' << tip.interface();
}

std::istream & tip_read_ascii(tree_interface_pair & tip, std::istream & input) {
	uint16_t t;
	uint16_t i;
	if (input >> t >> i)
		tip.assign(t, i);
	return input;
}
