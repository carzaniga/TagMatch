#ifndef FILTER_H_INCLUDED
#define FILTER_H_INCLUDED

#include <iostream>
#include <string>
#include <vector>

class filter {
public:
    typedef unsigned char pos_t;
    typedef std::vector<pos_t>::iterator iterator;
    typedef std::vector<pos_t>::const_iterator const_iterator;
    //static const unsigned int FILTER_SIZE = 192;
    static const unsigned int FILTER_SIZE = 192;

private:
    std::vector<pos_t> elements;

public:
    const_iterator begin() const {
		return elements.begin();
    }

    iterator begin() {
		return elements.begin();
    }

    const_iterator end() const {
		return elements.end();
    }

    iterator end() {
		return elements.end();
    }

    int count() const {
		return elements.size();
    }

    filter(): elements() { }

    filter(const filter & f): elements(f.elements) { }

    filter(const std::string & s) throw(int);
    filter & operator=(const std::string & s) throw(int);

    std::ostream & print(std::ostream & os) const;

    bool operator>=(const filter &b) const;
    
    void clear (){
        elements.clear();
    }
};

// extern std::istream & operator >> (std::istream & is, bloom_filter & f) throw (int); 

inline std::ostream & operator << (std::ostream & os, const filter & f) {
    return f.print(os);
}

#endif
