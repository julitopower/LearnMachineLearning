#include "sdl.hpp"
#include <chrono>
#include <ratio>
#include <thread>

int main(int, char **) {
  auto& w = viz::Window::get();
  w.clear();
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}
