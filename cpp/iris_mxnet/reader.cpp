#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <thread>

#include <mxnet-cpp/MxNetCpp.h>

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

int main(int argc, char** argv)
{
  namespace mx = mxnet::cpp;
  int batch_size {5};
  mx::index_t dim {4};  

  //  while (batch_size--) {
  std::cout << "Little pause 1" << std::endl;    
  auto train_iter = mx::MXDataIter("CSVIter")
    .SetParam("data_csv", "./iris_train_data.csv")
    .SetParam("label_csv", "./iris_train_label.csv")
    .SetParam("data_shape", mx::Shape{dim})
    .SetParam("label_shape", mx::Shape{1})    
    .SetParam("batch_size", batch_size)
    .CreateDataIter();

  //  std::cout << "Little pause " << std::endl;
  auto test_iter = mx::MXDataIter("CSVIter")
    .SetParam("data_csv", "./iris_test_data.csv")
    .SetParam("label_csv", "./iris_test_label.csv")
    .SetParam("data_shape", mx::Shape{dim})
    .SetParam("label_shape", mx::Shape{1})        
    .SetParam("batch_size", batch_size)        
    .CreateDataIter();


  //  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  
  //  }

  auto epochs = 200U;
  while(epochs--) {
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
    }
    test_iter.Reset();
    while (test_iter.Next()) {
      auto data_batch = test_iter.GetDataBatch();
    }
  }
  return 0;
}
