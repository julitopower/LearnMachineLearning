#include <csignal>
#include <iostream>
#include <unistd.h>

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

void set_signal_handlers() {
  void (*handler)(int) = [](int signal) {
    constexpr int MAX_FRAMES = 100;
    void *array[MAX_FRAMES];
    const std::size_t size = backtrace(array, MAX_FRAMES);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", signal);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
  };

  std::signal(SIGSEGV, handler);    
}

int main(int argc, char**argv)
{
  namespace mx = mxnet::cpp;

  // Set the termination handler
  set_terminate();
  set_signal_handlers();

  std::uint32_t dims = 4;
  std::uint32_t batch_size = 5;
  
  // Only provide mandatory arguments. Defaults work well
  MlpParams params{dims, {24,3}, batch_size};
  Mlp mlp{params};
  mlp.fit("./iris_train_data.csv", "./iris_train_label.csv",
	  "./iris_test_data.csv", "./iris_test_label.csv");

  mlp.save_model("/tmp/");
  Mlp mlp2{"/tmp"};
  mlp2.predict("./iris_train_data.csv");
  
  return 0;
}
