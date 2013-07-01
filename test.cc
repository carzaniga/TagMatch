#include <iostream>
#include <vector>
using namespace std;

class node {
public:
    unsigned char pos;
    unsigned char treeMask;
    bool set;
    node * t;
    node * f;
};
class end_node: public node{
public:
	vector<long> v;// a[3];
    vector<long [3]> ;
	//bool status;
	end_node():node(){};
	void settrue(bool a){
	//	status=a;
		set=1;
	}
};
int main() {
  cout<<sizeof(node)<<endl;
  cout<<sizeof(end_node)<<endl<<endl;
  
  node *p1 = new node();
  p1->set=1; 
  end_node *p2 = new end_node();
  p2->set=0;
  p2->settrue(1);
	
  std::cout << p1->set << std::endl;
  std::cout << p2->set << std::endl;
  node *p3= p2;
  std::cout << p3->set << std::endl;
  return 0;
}
