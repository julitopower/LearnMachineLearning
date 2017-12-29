#include <iostream>

#include "mlp.hpp"

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

  std::uint32_t dims = 4;
  std::uint32_t batch_size = 5;
  
  // Only provide mandatory arguments. Defaults work well
  MlpParams params{dims, {24,3}, batch_size};
  Mlp mlp{params};
  mlp.fit("./iris_train_data.csv", "./iris_train_label.csv",
	  "./iris_test_data.csv", "./iris_test_label.csv");
  
  return 0;
}
