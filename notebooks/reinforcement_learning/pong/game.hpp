#pragma once

#include <cstdlib>
#include <ctime>
#include <map>
#include <typeinfo>
#include <vector>

/*!
 * These are the elements necessary to define a very rudimentary
 * pong-lite game. In fact the is only one Racket, and a Ball.
 * The Ball is launched from the right of the screen with a random
 * velocity, and the player can control the Racket, up and down.
 *
 * The Game finishes when either the Ball collides with the Racket, or
 * goes behond the racket. The boundaries of the game cause the Ball
 * to bounce.
 *
 * The entire simulation is controlled with the Game class, which allows
 * advancing the simulation one step at a time, with a player action. The
 * Game also tracks a reward value, which is 1 in the case of collision
 * between the Ball and the Racket, or -1 if the Ball goes beyond the
 * Racket.
 */

class Ball {
public:
  Ball(int x, int y, int vx, int vy) : x{x}, y{y}, vx{vx}, vy{vy} {}

  std::pair<int, int> project(int time_delta) const {
    return {x + vx * time_delta, y + vy * time_delta};
  }

  void update(int time_delta) {
    auto values = project(time_delta);
    x = values.first;
    y = values.second;
  }

  int x, y, vx, vy;
  const int r = 3;
};

class Racket {
public:
  Racket(int x, int y, int vx, int vy) : x{x}, y{y}, vx{vx}, vy{vy} {}

  std::pair<int, int> project(int time_delta) const {
    return {x + vx * time_delta, y + vy * time_delta};
  }

  void update(int time_delta) {
    auto values = project(time_delta);
    x = values.first;
    y = values.second;
  }

  int x, y, vx, vy;
  int w = 3;
  int h = 10;
};

enum class Action { UP, DOWN, NOOP };

class Game {
public:
  Game() : racket_{0, 0, 0, 0}, ball_{0, 0, 0, 0} { srand(time(nullptr)); }
  void reset() {
    ball_.vx = 0;
    while (ball_.vx == 0) {
      ball_.vx = -10 * static_cast<double>(rand()) / RAND_MAX;
    }
    ball_.vy = -10 * static_cast<double>(rand()) / RAND_MAX;

    racket_.vx = 0;
    racket_.vy = 0;
    ball_.x = width_;
    ball_.y = height_ / 2;
    racket_.x = 0;
    racket_.y = height_ / 2;
    reward_ = 0;
  }

  void step(Action action) {
    // Update ball and racket position
    ball_.update(1);

    // Make sure the ball stays in the court boundaries
    // We will want to add some randomness here, but not yet
    if (ball_.y + ball_.r >= width_ || ball_.y - ball_.r <= 0) {
      ball_.vy *= -1;
    }

    switch (action) {
    case Action::UP:
      racket_.vy = 2;
      break;
    case Action::DOWN:
      racket_.vy = -2;
      break;
    default:
      break;
    }
    racket_.update(1);

    // Check for collision AABB
    // Rightmost edge of Ball is always on the right with respect of the racket
    // leftmost edge
    if (racket_.x + racket_.w > ball_.x - ball_.r &&
        racket_.y - racket_.h < ball_.y + ball_.r &&
        racket_.y + racket_.h > ball_.y - ball_.r) {
      // Collision!
      reward_ = 1;
    } else if (ball_.x <= 0) {
      reward_ = -1;
    }
  }

  int reward() const { return reward_; }

  std::vector<int> state() {
    return {ball_.x,   ball_.y,     ball_.vx,   ball_.vy, racket_.x,
            racket_.y, racket_.vx, racket_.vy, reward_};
  }

private:
  Racket racket_;
  Ball ball_;
  int width_ = 800;
  int height_ = 600;
  int reward_ = 0;
};
