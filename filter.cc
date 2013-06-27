#include <vector>
#include <iostream>

#include "filter.h"

filter::filter(const std::string & s) throw (int) {
    *this = s;
}

filter & filter::operator=(const std::string & s) throw (int) {
    elements.clear();
    pos_t p = 0;
    int k=0;
    for(std::string::const_iterator c = s.begin(); c != s.end(); ++c) {
	switch (*c) {
	case '1': 
		  k++;
		  if(k<15)
			elements.push_back(p);
	case '0': ++p; 
	    break;
	default:
	    throw -1;
	}
    }
    return *this;
}

#if 0
std::ostream & filter::print(std::ostream & os) const {
    const_iterator e = elements.begin();
    for(pos_t p = 0; p < FILTER_SIZE; ++p) {
	if (e != elements.end() && *e == p) {
	    os << '1';
	    ++e;
	} else {
	    os << '0';
	}
    }
    return os;
}
#else
std::ostream & filter::print(std::ostream & os) const {
    for(const_iterator e = elements.begin(); e != elements.end(); ++e)
	os << ' ' << (unsigned int)*e;
    return os;
}
#endif

bool filter::operator>=(const filter & b) const {
    std::vector<pos_t>::const_iterator eb = b.elements.begin();
    if (eb == b.elements.end())
	return true;

    std::vector<pos_t>::const_iterator ea = elements.begin();
    if (ea == elements.end())
	return false;

    for(;;) {
	if (*ea > *eb) {
	    ++eb;
	    if (eb == b.elements.end())
		return true;
	} else if (*ea < *eb) {
	    ++ea;
	    if (ea == elements.end())
		return false;
	} else {
	    ++eb;
	    if (eb == b.elements.end())
		return true;
	    ++ea;
	    if (ea == elements.end())
		return false;
	}
    }
}
