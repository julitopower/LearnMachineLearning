#pragma once

#include <cstdlib>
#include <ctime>
#include <map>
#include <typeinfo>
#include <vector>

/*!
 * These are the elements necessary to define a very simplistic
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
 *
 * There are two sets of coordinates. Ball and Racket represent real world
 * coordinates as integers, and I assume they are in centimiters. The
 * Game class maps coordinates and velocities to a resolution expected
 * by the agent. For instance, we may have a world of 2m x 1m, but the agent
 * is trained on a 255 x 255 grid.
 */

/*! \brief A ball in our pong game */
struct Ball {
  /*!
   * \param x Real world x coordinate in cm
   * \param y Real world y coordinate in cm
   * \param vx Real world velocity in the x axis in cm/step
   * \param vy Real world velocity in the y axis in cm/step
   */
  Ball(int x, int y, int vx, int vy) : x{x}, y{y}, vx{vx}, vy{vy} {}

  /*! \brief Project the state in time time_delta steps */
  std::pair<int, int> project(int time_delta) const {
    return {x + vx * time_delta, y + vy * time_delta};
  }

  void update(int time_delta) {
    auto values = project(time_delta);
    x = values.first;
    y = values.second;
  }

  // Real world coordinates in cm
  int x, y, vx, vy;
  // Radius in cm
  const int r = 3;
};

class Racket {
public:
  Racket(int x, int y, int vx, int vy, int w, int h)
      : x{x}, y{y}, vx{vx}, vy{vy}, w{w}, h{h} {}

  std::pair<int, int> project(int time_delta) const {
    return {x + vx * time_delta, y + vy * time_delta};
  }

  void update(int time_delta) {
    auto values = project(time_delta);
    x = values.first;
    y = values.second;
  }

  // Real world coordinates in cm. Represents the center of the racket
  // The racket corners are calculated from the center, adding/subtracting
  // widht and hight
  int x, y, vx, vy;
  // Width in cm: This is actually total_width / 2.
  int w;
  // Hight in cm: This is actually total_hight / 2.
  int h;
};

// Enum to capture the player actions
enum class Action { UP, DOWN, NOOP };

class Game {
public:
  /*! \brief Simulation of a Pong game
   *
   * \param width The world width of the court in cm
   * \param height The world height of the court in cm
   * \param viewport_x The RL court width (normally smaller than the world one)
   * \param viewport_y The RL court height (normally smaller than the world one)
   *
   * Set the size of the racket as a function of the real world width and
   * height. For instance, we may want the racket height to be 10% of the court
   * height.
   */
  Game(int width, int height, int viewport_x, int viewport_y)
      : racket_{0, 0, 0, 0, 3, static_cast<int>(height * .05)},
        ball_{0, 0, 0, 0}, width_{width}, height_{height},
        proj_width_{viewport_x}, proj_height_{viewport_y} {
    srand(time(nullptr));
  }

  void reset() {
    // Set random ball velocities
    ball_.vx = 0;
    while (ball_.vx == 0) {
      ball_.vx = -width_ * 0.005 * static_cast<double>(rand()) / RAND_MAX;
    }

    ball_.vy = -height_ * 0.05 * static_cast<double>(rand()) / RAND_MAX;

    // Position Racket and Ball in initial positions
    racket_.vx = 0;
    racket_.vy = 0;
    ball_.x = width_;
    ball_.y = height_ / 2;
    racket_.x = 0;
    racket_.y = height_ / 2;

    // Reset reward and completion flags
    reward_ = 0;
    done_ = false;
  }

  void step(Action action) {
    // Update ball and racket position
    ball_.update(1);

    // Make sure the ball stays in the court boundaries
    // We will want to add some randomness here, but not yet
    if (ball_.y + ball_.r >= height_ || ball_.y - ball_.r <= 0) {
      ball_.vy *= -1;
    }

    switch (action) {
    case Action::UP:
      racket_.vy = height_ * 0.025;
      break;
    case Action::DOWN:
      racket_.vy = -height_ * 0.025;
      break;
    default:
      racket_.vy = 0;
      break;
    }
    std::size_t y = racket_.y;
    racket_.update(1);
    // Make sure the racket statys in the court
    if (racket_.y + racket_.h >= height_ || racket_.y - racket_.h <= 0) {
      racket_.y = y;
    }

    // Check for collision AABB
    // Rightmost edge of Ball is always on the right with respect of the racket
    // leftmost edge
    if (racket_.x + racket_.w > ball_.x - ball_.r &&
        racket_.y - racket_.h < ball_.y + ball_.r &&
        racket_.y + racket_.h > ball_.y - ball_.r) {
      // Collision!
      reward_ = 1;
      done_ = true;
    } else if (ball_.x <= 0) {
      reward_ = -1;
      done_ = true;
    }
  }

  int reward() const { return reward_; }

  bool done() const { return done_; }

  const std::vector<double> &state() {
    state_[0] = ball_.x * xfactor;
    state_[1] = ball_.y * yfactor;
    state_[2] = ball_.vx * xfactor;
    state_[3] = ball_.vy * yfactor;
    state_[4] = racket_.x * xfactor;
    state_[5] = racket_.y * yfactor;
    state_[6] = racket_.vx * xfactor;
    state_[7] = racket_.vy * yfactor;
    return state_;
  }

private:
  Racket racket_;
  Ball ball_;
  const int width_;
  const int height_;
  const int proj_width_;
  const int proj_height_;
  const double xfactor = proj_width_ / static_cast<double>(width_);
  const double yfactor = proj_height_ / static_cast<double>(height_);
  int reward_ = 0;
  std::vector<double> state_ = std::vector<double>(8, 0);
  bool done_ = false;
};

extern "C" {
typedef void *PongHdlr;

PongHdlr pong_new(int w, int h, int pw, int ph) {
  return static_cast<void *>(new Game{w, h, pw, ph});
}

void pong_delete(PongHdlr pong) { delete static_cast<Game *>(pong); }

void pong_step(PongHdlr pong, int action) {
  auto &game = *static_cast<Game *>(pong);
  switch (action) {
  case 0:
    game.step(Action::NOOP);
    break;
  case 1:
    game.step(Action::UP);
    break;
  case 2:
    game.step(Action::DOWN);
    break;
  default:
    break;
  }
}

const double *pong_state(PongHdlr pong) {
  auto &game = *static_cast<Game *>(pong);
  return game.state().data();
}

void pong_reset(PongHdlr pong) {
  auto &game = *static_cast<Game *>(pong);
  game.reset();
}

int pong_reward(PongHdlr pong) {
  auto &game = *static_cast<Game *>(pong);
  return game.reward();
}

int pong_done(PongHdlr pong) {
  auto &game = *static_cast<Game *>(pong);
  return game.done();
}
}
