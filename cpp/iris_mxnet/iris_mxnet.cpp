#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <thread>

#include <mxnet-cpp/MxNetCpp.h>

/*
 Termination handler
*/
void set_terminate() {
  std::set_terminate([]() {
      std::cout << "Teminating application" << std::endl;
      auto ex = std::current_exception();
      try {
	std::rethrow_exception(ex);
      } catch(const std::exception& ex) {
	std::cout << ex.what() << std::endl;
      }
      std::exit(1);      
    });
}

int main(int argc, char**argv)
{
  namespace mx = mxnet::cpp;

  // Set the termination handler
  set_terminate();

  // Let's define the network first
  auto data = mx::Symbol::Variable("data");
  auto label = mx::Symbol::Variable("labels");


  // First hidden layer, fully connected
  auto weight1 = mx::Symbol::Variable("w1");
  auto bias1 = mx::Symbol::Variable("b1");  
  auto fc1 = mx::FullyConnected(data, weight1, bias1, 24);
  auto ac1 = mx::Activation(fc1, mx::ActivationActType::kRelu);

  // Second hidden layer, fully connected but without activation function
  auto weight2 = mx::Symbol::Variable("w2");
  auto bias2 = mx::Symbol::Variable("b2");    
  auto fc2 = mx::FullyConnected(ac1, weight2, bias2, 3);

  // Output layer
  auto net = mx::SoftmaxOutput(fc2, label);

  ////////////////////////////////////////////////////////////////////////////////
  // Now that the network has been built, we need to configure it
  ////////////////////////////////////////////////////////////////////////////////
  std::map<std::string, mx::NDArray> args;
  auto batch_size {5};
  auto learning_rate {0.01d};
  auto weight_decay {0.001d};
  mx::index_t dim {4};
  auto ctx = mx::Context::cpu();
  args["data"] = mx::NDArray(mx::Shape(batch_size, dim), ctx);
  args["labels"] = mx::NDArray(mx::Shape(batch_size), ctx);

  // Let MXNet infer shapes other parameters such as weights
  net.InferArgsMap(ctx, &args, args);

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  auto initializer = mx::Uniform(0.01);
  for (auto& arg : args) {
    // arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }

  // Create sgd optimizer
  auto opt = mx::OptimizerRegistry::Find("sgd");
  opt->SetParam("lr", learning_rate)
    ->SetParam("wd", weight_decay);

  // Create executor by binding parameters to the model
  auto *exec = net.SimpleBind(ctx, args);

  // Let's load the data
  auto train_iter = mx::MXDataIter("CSVIter")
    .SetParam("data_csv", "./iris_train_data.csv")
    .SetParam("label_csv", "./iris_train_label.csv")
    .SetParam("data_shape", mx::Shape{dim})
    .SetParam("label_shape", mx::Shape{1})    
    .SetParam("batch_size", batch_size)
    .CreateDataIter();

  auto test_iter = mx::MXDataIter("CSVIter")
    .SetParam("data_csv", "./iris_test_data.csv")
    .SetParam("label_csv", "./iris_test_label.csv")
    .SetParam("data_shape", mx::Shape{dim})
    .SetParam("label_shape", mx::Shape{1})        
    .SetParam("batch_size", batch_size)        
    .CreateDataIter();

  // There is a bug in MxNet prefetcher, sleeping
  // for a little while seems to avoid it
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  
  auto arg_names = net.ListArguments();
  auto epochs = 200;
  auto tic = std::chrono::system_clock::now();  
  while (epochs > 0) {
    train_iter.Reset();

    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      // Set data and label
      data_batch.data.CopyTo(&args["data"]);
      data_batch.label.CopyTo(&args["labels"]);
      args["data"].WaitAll();
      args["labels"].WaitAll();      
      // Compute gradients
      exec->Forward(true);
      exec->Backward();
      // Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
	if (arg_names[i] == "data" || arg_names[i] == "labels") continue;
	opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }
    }

    mx::Accuracy acc;
    
    test_iter.Reset();
    while (test_iter.Next()) {
      auto data_batch = test_iter.GetDataBatch();
      data_batch.data.CopyTo(&args["data"]);
      data_batch.label.CopyTo(&args["labels"]);
      // Forward pass is enough as no gradient is needed when evaluating
      exec->Forward(false);
      acc.Update(data_batch.label, exec->outputs[0]);
    }

    std::cout << "Epoch: " << epochs << " " << acc.Get() << std::endl;;
    --epochs;
  }

  auto toc = std::chrono::system_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(toc  - tic).count() << std::endl;  
  
  return 0;
}
