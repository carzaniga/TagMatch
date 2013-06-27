#include <cstddef>
#include <iostream>

static const unsigned int ALIGNMENT 
= offsetof(
	   struct {
	     char x; 
	     union {
	       char c;
	       int i;
	       long l;
	       bool b;
	       double d;
	       // long double ld;
	       long long ll;
	       void* vp;
	       void(*vfp)(void);
	     } y;},
	   y);

class node {
public:
    unsigned char pos;
    unsigned char treeMask;
    bool set;
    node * t;
    node * f;
};

int main() {
  std::cout << ALIGNMENT << std::endl;
  node A[2];

  std::cout << sizeof(A) << std::endl;
  
  void * p1 = new node();
  void * p2 = new node();

  std::cout << p2 << std::endl;
  std::cout << p1 << std::endl;

  for(int i = 1; i < 100; ++i) {
    p1 = new char[3];
    p2 = new node();
    std::cout << p2 << ' ' << p1 << ' ' << (reinterpret_cast<char*>(p2)-reinterpret_cast<char*>(p1)) << std::endl;
  }
  return 0;
}
