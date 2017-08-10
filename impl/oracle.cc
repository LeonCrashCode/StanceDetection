#include "impl/oracle.h"

#include <cassert>
#include <fstream>
#include <strstream>

#include "dynet/dict.h"

using namespace std;

namespace stance {


Oracle::~Oracle() {}

void Oracle::load_oracle(const string& file) {
  cerr << "Loading oracle from " << file <<"\n";
  ifstream in(file.c_str());
  assert(in);
  string line;
  while(getline(in, line)) {
    sents.resize(sents.size() + 1);
    auto& cur_sent = sents.back();
    istrstream istr(line.c_str());
    string word;
    while(istr>>word) {
      if(word == "|||") break;
      cur_sent.target.push_back(d->convert(word));
    }
    while(istr>>word) {
      if(word == "|||") break;
      cur_sent.sent.push_back(d->convert(word));
    }
    istr >> word;
    labels.push_back(ld->convert(word));
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative       label vocab size: " << ld->size() << endl;
}
} // namespace stance
