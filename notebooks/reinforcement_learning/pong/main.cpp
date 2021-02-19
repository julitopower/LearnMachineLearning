#include "game.hpp"
#include <iostream>

namespace {
template <typename T>
void p(T v) {
  for (const auto &value : v) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;
}
} // unnamed namespace

int main(int argc, char *argv[]) {
  Game g{2000, 1000, 255, 255};


  while (g.reward() != 1) {
    g.reset();    
    while (g.reward() == 0) {
      g.step(Action::NOOP);
    }
    p(g.state());
  }

  return 0;
}
