#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>

#include <mxnet-cpp/MxNetCpp.h>

struct MlpParams {
  MlpParams() :
    dim{0},
    hidden{},
    batch_size{0},
    learning_rate{0},
    weight_decay{0},
    epochs{0} {};
  
  MlpParams(std::uint32_t dim, const std::vector<std::size_t>& hidden, std::size_t batch_size = 10,
	    double learning_rate = 0.01d, double weight_decay = 0.001d, std::size_t epochs = 200) :
    dim{dim},
    hidden{hidden},
    batch_size{batch_size},
    learning_rate{learning_rate},
    weight_decay{weight_decay},
    epochs{epochs} {};

  // Feature dimensions in the output
  const std::uint32_t dim;

  // Number of nodes per hiddent layer. The last value is the number
  // of inputs to the output layer, and it won't be subjected to the
  // activation function
  const std::vector<std::size_t> hidden;

  // Minibatch size
  const std::size_t batch_size;

  // Learning rate
  const double learning_rate;

  // weight_decay
  const double weight_decay;

  // Number of epochs
  std::size_t epochs;
};

/*
  Multilayer Perceptron.

  Uses Relu as activation function in hidden layers, and Softmax in
  output layer.
*/
class Mlp {
public:
  Mlp(const MlpParams& params);

  Mlp(const std::string& dir);
  
  /*
    If test data is provided, after each epoch the accurary in the
    test dataset is printed
  */
  void fit(const std::string& data_csv_path,
	   const std::string& labels_csv_path,
	   const std::string& data_test_path = "",
	   const std::string& labels_test_path = "");
  /*
    Not implemented yet
  */
  void predict(const std::string& filepath);

  void save_model(const std::string& dir);

  /*
     Not implemented yet
  */
  void reset();

private:
  MlpParams params_;
  std::vector<mxnet::cpp::Symbol> weights_;
  std::vector<mxnet::cpp::Symbol> biases_;
  std::vector<mxnet::cpp::Symbol> outputs_;
  mxnet::cpp::Symbol network_;
  std::map<std::string, mxnet::cpp::NDArray> args_;
  mxnet::cpp::Optimizer* opt_;
  mxnet::cpp::Executor* exec_;
};

#endif /* MLP_H */
