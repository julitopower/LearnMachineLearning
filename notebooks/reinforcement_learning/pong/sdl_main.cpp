#include "game.hpp"
#include "sdl.hpp"
#include <chrono>
#include <iostream>
#include <ratio>
#include <thread>

int main(int, char **) {
  auto &w = viz::Window::get();
  Game g{800, 600, 800, 600};
  g.reset();
  while (!g.done()) {
    auto &ball = g.ball();
    auto &racket = g.racket();
    w.clear();
    w.add_rect(0, 0, 800, 600);
    w.add_rect(1, 1, 798, 598);
    w.fill_rect(ball.x, ball.y, ball.r, ball.r);
    w.fill_rect(racket.x, racket.y, racket.w, racket.h);
    w.present();
    g.step(Action::NOOP);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}
