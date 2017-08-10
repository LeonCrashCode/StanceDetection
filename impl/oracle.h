#ifndef STANCE_ORACLE_H_
#define STANCE_ORACLE_H_

#include <iostream>
#include <vector>
#include <string>

namespace dynet { class Dict; }

namespace stance {

struct Sentence {
  bool SizesMatch() const { return true; }
  size_t size() const { return sent.size(); }
  std::vector<int> sent;
  std::vector<int> target;
};

// base class for transition based parse oracles
struct Oracle {

public:
  ~Oracle();
  Oracle(dynet::Dict* dict, dynet::Dict* ldict) : d(dict), ld(ldict), sents() {}
  unsigned size() const { return sents.size(); }
  dynet::Dict* d;  // dictionary of terminal symbols
  dynet::Dict* ld; // dictionary of action types
  std::string devdata;
  std::vector<Sentence> sents;
  std::vector<int> labels;
  void load_oracle(const std::string& line);
};

} // namespace stance

#endif
