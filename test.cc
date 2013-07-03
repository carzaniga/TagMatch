#include <iostream>
#include <vector>
#include <bitset>
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
    //vector<long [3]> ;
	//bool status;
	end_node():node(){};
	void settrue(bool a){
	//	status=a;
		set=1;
	}
};

struct mypairs{
long c;//  *tete;
long b;//*queue;
long a;
};

int main() {
  cout<<sizeof(node)<<endl;
  cout<<sizeof(end_node)<<endl;
  cout<<sizeof(mypairs)<<endl<<endl;
  cout<<"size of different vectors="<<sizeof(vector<int>)<<"\t"<<sizeof(vector<unsigned char>)<<endl;
  mypairs mp;
  mp.a=23;
  mp.b=53;
  mp.c=123;
  mypairs *myp;
  myp=&mp;
  cout<<(*myp).c<<endl;
  typedef long mytype[3];
  mytype *krs;
  (*krs)[1]=764;
  cout<<"size of mytype "<<sizeof(krs)<<"\t"<<(*krs)[1]<<endl;
  vector<mytype> myv;
  cout<<sizeof(myv)<<endl;
  bitset<192> *bs;
  cout<<sizeof(*bs)<<endl; 

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
