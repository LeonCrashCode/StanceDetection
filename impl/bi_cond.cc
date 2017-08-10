#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>

#include <unordered_map>
#include <unordered_set>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/io.h"
#include "dynet/dict.h"

#include "impl/oracle.h"
#include "impl/cl-args.h"

dynet::Dict termdict, labeldict;

volatile bool requested_stop = false;
unsigned VOCAB_SIZE = 0;
unsigned LABEL_SIZE = 0;

using namespace dynet;
using namespace std;

Params params;

unordered_map<unsigned, vector<float>> pretrained;

struct StanceBuilder {

  LSTMBuilder l2rbuilder;
  LSTMBuilder r2lbuilder;
  LSTMBuilder target_l2rbuilder;
  LSTMBuilder target_r2lbuilder;
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // input action embeddings
  
  Parameter p_bias;
  Parameter p_sent2h;
  
  Parameter p_lbias;
  Parameter p_h2l;
  explicit StanceBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      l2rbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      r2lbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      target_l2rbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      target_r2lbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_bias(model->add_parameters({params.nonlinear_dim})),
      p_sent2h(model->add_parameters({params.nonlinear_dim, params.bilstm_hidden_dim*2})),
      p_lbias(model->add_parameters({LABEL_SIZE})),
      p_h2l(model->add_parameters({LABEL_SIZE, params.nonlinear_dim})){
      for (auto it : pretrained){
        p_w.initialize(it.first, it.second);
        p_t.initialize(it.first, it.second);
      }
    }
Expression log_prob_detector(ComputationGraph* hg,
		     const stance::Sentence& sent,
                     const int& label,
                     vector<int>& eval,
		     int* pred,
		     bool train) {
if(params.debug) cerr<<"sent size: "<<sent.sent.size()<<", target size: "<<sent.target.size()<<"\n";

    vector<Expression> input_expr(sent.sent.size());
    for (unsigned i = 0; i < sent.sent.size(); ++i) {
      int wordid = sent.sent[i];
if(params.debug) cerr<<termdict.convert(wordid)<<"\n"; 
      input_expr[i] = lookup(*hg, p_w, wordid);
      if(train) input_expr[i] = dropout(input_expr[i], params.pdrop);
    }
   
    vector<Expression> target_expr(sent.target.size());
    for (unsigned i = 0; i < sent.target.size(); ++i) {
      int wordid = sent.target[i];
if(params.debug) cerr<<termdict.convert(wordid)<<"\n";
      target_expr[i] = lookup(*hg, p_t, wordid);
      if(train) target_expr[i] = dropout(target_expr[i], params.pdrop);
    }

    target_l2rbuilder.new_graph(*hg);
    target_r2lbuilder.new_graph(*hg);
    target_l2rbuilder.start_new_sequence();
    target_r2lbuilder.start_new_sequence();

/*    if(train){
	target_l2rbuilder.set_dropout(params.pdrop);
        target_r2lbuilder.set_dropout(params.pdrop);
    }
*/
    for (unsigned i = 0; i < target_expr.size(); i ++) {
      target_l2rbuilder.add_input(target_expr[i]);
      target_r2lbuilder.add_input(target_expr[target_expr.size()-i-1]);
    }

    vector<Expression> l2r_cond = target_l2rbuilder.final_s();
    vector<Expression> r2l_cond = target_r2lbuilder.final_s();

    l2rbuilder.new_graph(*hg);
    r2lbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence(l2r_cond);
    r2lbuilder.start_new_sequence(r2l_cond);

/*    if(train){
        l2rbuilder.set_dropout(params.pdrop);
        r2lbuilder.set_dropout(params.pdrop);
    }
*/
    for(unsigned i = 0; i < input_expr.size(); i ++){
      l2rbuilder.add_input(input_expr[i]);
      r2lbuilder.add_input(input_expr[input_expr.size()-i-1]);
    }

    Expression i_bias = parameter(*hg, p_bias);
    Expression i_sent2h = parameter(*hg, p_sent2h);
    
    Expression i_lbias = parameter(*hg, p_lbias);
    Expression i_h2l = parameter(*hg, p_h2l);

    Expression non_linear = tanh(affine_transform({i_bias, i_sent2h, concatenate({l2rbuilder.back(), r2lbuilder.back()})}));
    Expression diste = affine_transform({i_lbias, i_h2l, non_linear});
    Expression log_diste = log_softmax(diste);
    vector<float> log_dist = as_vector(hg->incremental_forward(log_diste));
    float best_score = log_dist[0];
    int best_l = 0;
    for(unsigned i = 1; i < log_dist.size(); i ++) {
      if(log_dist[i] > best_score) {
        best_score = log_dist[i];
        best_l = (int)i;
      }
    }
if(params.debug)	std::cerr<<"best label "<<best_l<<" " << labeldict.convert(best_l)<<"\n";
    if(labeldict.convert(best_l) == "FAVOR") {
      eval[1] += 1;
      if(best_l == label) eval[0] += 1;
    }
    else if(labeldict.convert(best_l) == "AGAINST"){
      eval[4] += 1;
      if(best_l == label) eval[3] += 1;
    }

    if(labeldict.convert(label) == "FAVOR") eval[2] += 1;
    else if(labeldict.convert(label) == "AGAINST") eval[5] += 1;

    if (pred) (*pred) = best_l;
    Expression tot_neglogprob = -pick(log_diste, label);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;
  }
};

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

void get_eval(const vector<int> eval,
		double& fp, double& fr, double& ff,
		double& ap, double& ar, double& af,
		double& mf){

	assert(eval.size() == 6);
	fp = (eval[1] == 0 ? 0 : eval[0]*1.0 / eval[1]);
	fr = (eval[2] == 0 ? 0 : eval[0]*1.0 / eval[2]);
        ff = ((fp == 0 || fr == 0) ? 0 : 2*fp*fr / (fp+fr));

        ap = (eval[4] == 0 ? 0 : eval[3]*1.0 / eval[4]);
        ar = (eval[5] == 0 ? 0 : eval[3]*1.0 / eval[5]);
        af = ((ap == 0 || ar == 0) ? 0 : 2*ap*ar / (ap+ar));

	mf = (ff + af) / 2;
}

int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 25;

  get_args(argc, argv, params);

  ostringstream os;
  os << "detector" 
     << '_' << params.layers
     << '_' << params.input_dim
     << '_' << params.bilstm_input_dim
     << '_' << params.bilstm_hidden_dim
     << '_' << params.nonlinear_dim
     << "-pid" << getpid() << ".params";

  double best_acc = 0;
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

//=====================================================================================================================

  stance::Oracle corpus(&termdict, &labeldict);
  corpus.load_oracle(params.train_file);

  if (params.words_file != "") {
    cerr << "Loading from " << params.words_file << " with" << params.pretrained_dim << " dimensions\n";
    ifstream in(params.words_file.c_str());
    string line;
    getline(in, line);
    vector<float> v(params.pretrained_dim, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < params.pretrained_dim; ++i) lin >> v[i];
      unsigned id = termdict.convert(word);
      pretrained[id] = v;
    }
  }

  termdict.freeze();
  termdict.set_unk("UNK");
  labeldict.freeze();

  VOCAB_SIZE = termdict.size();
  LABEL_SIZE = labeldict.size();

  stance::Oracle dev_corpus(&termdict, &labeldict);
  stance::Oracle test_corpus(&termdict, &labeldict);
  if(params.dev_file != "") dev_corpus.load_oracle(params.dev_file);
  if(params.test_file != "") test_corpus.load_oracle(params.test_file);

//==========================================================================================================================
  
  Model model;
  StanceBuilder detector(&model, pretrained);
  if (params.model_file != "") {
    TextFileLoader loader(params.model_file);
    loader.populate(model);
  }
  
  //TRAINING
  if (params.train) {
    signal(SIGINT, signal_callback_handler);

    Trainer* sgd = NULL;
    unsigned method = params.train_methods;
    if(method == 0)
        sgd = new SimpleSGDTrainer(model, params.init_lr);
    else if(method == 1)
        sgd = new MomentumSGDTrainer(model);
    else if(method == 2){
        sgd = new AdagradTrainer(model);
//        sgd->clipping_enabled = false;
    }
    else if(method == 3){
        sgd = new AdamTrainer(model);
//        sgd->clipping_enabled = false;
    }

    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min(status_every_i_iterations, corpus.size());
    unsigned si = corpus.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.size() << endl;
    vector<int> eval;
    for(unsigned i = 0; i < 6; i ++){ eval.push_back(0);}
    double llh = 0;
    bool first = true;
    int iter = -1;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.size()) {
             si = 0;
             if (first) { first = false; } else {/* sgd->update_epoch();*/ sgd->learning_rate *= (1 - params.lr_decay); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
   	   const auto& sentence = corpus.sents[order[si]];
           const int& label = corpus.labels[order[si]];

	   ComputationGraph hg;
           Expression nll = detector.log_prob_detector(&hg, sentence, label, eval, NULL, true);

           double lp = as_scalar(hg.incremental_forward(nll));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward(nll);
           sgd->update();
           llh += lp;
           ++si;
      }
      sgd->status();
      
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);

      double favor_p, favor_r, favor_f;
      double against_p, against_r, against_f;
      double macro_f;
      
      get_eval(eval, favor_p, favor_r, favor_f, against_p, against_r, against_f, macro_f);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<")"
           << " per-sent-ppl: " << exp(llh / status_every_i_iterations)
           << " macro-F: " << macro_f
           << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;

      llh = 0;
      eval.clear();
      for(unsigned i = 0; i < 6; i ++){ eval.push_back(0);}

      static int logc = 0;
      ++logc;

      if (logc % 4 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
	double llh = 0;
        vector<int> eval;
        for(unsigned i = 0; i < 6; i ++){ eval.push_back(0);}
	auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
	   const auto& sentence = dev_corpus.sents[sii];
           const int& label = dev_corpus.labels[sii];

           ComputationGraph hg;
	   Expression nll = detector.log_prob_detector(&hg, sentence, label, eval, NULL, false);
           double lp = as_scalar(hg.incremental_forward(nll));
           llh += lp;
	}
        auto t_end = std::chrono::high_resolution_clock::now();

        double favor_p, favor_r, favor_f;
        double against_p, against_r, against_f;
        double macro_f;

        get_eval(eval, favor_p, favor_r, favor_f, against_p, against_r, against_f, macro_f);
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\t"
               	<<" llh: " << llh << " "
		<<" favor: " << favor_p << " " << favor_r << " " << favor_f<<" "
		<<" against: " << against_p << " " << against_r << " " << against_f<<" "
                <<" macro-f: " << macro_f
		<<"\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;

        if (macro_f > best_acc) {
          best_acc = macro_f;

      	  ostringstream part_os;
	  part_os << "detector"
     		<< '_' << params.layers
     		<< '_' << params.input_dim
     		<< '_' << params.bilstm_input_dim
     		<< '_' << params.bilstm_hidden_dim
                << '_' << params.nonlinear_dim
     		<< "-pid" << getpid()
		<< "-part" << (tot_seen/corpus.size()) << ".params";
	  const string part = part_os.str();
 
	  TextFileSaver saver("model/"+part);
	  saver.save(model);  
        }
      }
    }
    delete sgd;
  } // should do training?
  else{ // do test evaluation
	ofstream out("test.out");
        unsigned test_size = test_corpus.size();
        auto t_start = std::chrono::high_resolution_clock::now();
        double llh = 0;
        vector<int> eval;
        for(unsigned i = 0; i < 6; i ++){ eval.push_back(0);}
        for (unsigned sii = 0; sii < test_size; ++sii) {
                const auto& sentence=test_corpus.sents[sii];
                const int& label=test_corpus.labels[sii];
                ComputationGraph hg;
                int pred;
                Expression nll = detector.log_prob_detector(&hg, sentence, label, eval, &pred, false);
                double lp = as_scalar(hg.incremental_forward(nll));
                llh += lp;

        }
        auto t_end = std::chrono::high_resolution_clock::now();
	double favor_p, favor_r, favor_f;
        double against_p, against_r, against_f;
        double macro_f;

        get_eval(eval, favor_p, favor_r, favor_f, against_p, against_r, against_f, macro_f);

        cerr << "  TEST llh= " << llh
                <<" favor: " << favor_p << " " << favor_r << " " << favor_f<<" "
                <<" against: " << against_p << " " << against_r << " " << against_f<<" "
                <<" macro-f: " << macro_f
		<<"\t[" << test_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
  }
}

