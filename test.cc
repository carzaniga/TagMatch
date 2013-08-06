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
  static vector<int> my_vec;
  static int myval;

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
//		cout<<krs<<endl;
	}
};

struct mypairs{
long c;//  *tete;
long b;//*queue;
long a;
};

//int node::myval=12;
//int krs=15;
//static vector<node> nd_vec;
int main() {
	unsigned long ul=5;
	int lastpos=__builtin_clzl(ul);
	cout<<lastpos<<" "<<__builtin_clzl(1ul)<<endl;
//	cout<<"aa "<<nn.pos<<endl;
//	nd_vec.push_back(nn);
//	cout<<nd_vec.size()<<endl;
//	nd_vec.at(0).pos=52;
//	cout<<"asdasds"<<nd_vec.at(0).pos<<"+++"<<endl;
//	cout<<nn.pos<<endl;
//	unsigned short aa=5;
//	unsigned short bb=0;
//	aa<<2;
//	cout<<aa<<"\t"<<bb<<endl;
// bitset<5> *p = new bitset<5>(25);
//cout<<*p<<endl;
//
//  bitset<5> bsss(13);
//  bitset<5> bs2(17);
// cout<<bsss<<"\t"<<bs2<<endl;
//  bitset<5> temp(0);
//	temp=bsss;
//  temp&=bs2;
//  cout<<temp<<"\t"<<bs2<<"\t"<<bsss<<endl;
//  vector<int> myv2;
//  myv2.push_back(10);
//  cout<<myv2.size()<<endl;
//  cout<<node::myval<<endl;
//
//  cout<<sizeof(node)<<endl;
//  cout<<sizeof(end_node)<<endl;
//  cout<<sizeof(mypairs)<<endl<<endl;
//  cout<<"size of different vectors="<<sizeof(vector<int>)<<"\t"<<sizeof(vector<unsigned char>)<<endl;
//  mypairs mp;
//  mp.a=23;
//  mp.b=53;
//  mp.c=123;
//  mypairs *myp;
//  myp=&mp;
//  cout<<(*myp).c<<endl;
//  typedef long mytype[3];
//  mytype *krs;
//  (*krs)[1]=764;
//  cout<<"size of mytype "<<sizeof(krs)<<"\t"<<(*krs)[1]<<endl;
//  vector<mytype> myv;
//  cout<<sizeof(myv)<<endl;
//  bitset<192> *bs;
//  cout<<sizeof(*bs)<<endl; 
//
//  node *p1 = new node();
//  p1->set=1; 
//  end_node *p2 = new end_node();
//  p2->set=0;
//  p2->settrue(1);
//	
//  std::cout << p1->set << std::endl;
//  std::cout << p2->set << std::endl;
//  node *p3= p2;
//  std::cout << p3->set << std::endl;
  return 0;
}
