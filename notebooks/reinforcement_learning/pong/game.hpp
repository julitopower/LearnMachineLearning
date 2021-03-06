#pragma once

#include <cstdlib>
#include <ctime>
#include <map>
#include <typeinfo>
#include <vector>

#include "sdl.hpp"

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
   * Coordinates indicate a corner of the squared ball. The ball extends to
   * (x + width, y + hight)
   *
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
  // TODO: Don't hardcode the radius. Radius in cm
  const int r = 10;
};

class Racket {
public:
  /*!
   * Coordinates indicate a corner of the squared racket. The ball extends to
   * (x + width, y + hight)
   *
   * \param x Real world x coordinate in cm
   * \param y Real world y coordinate in cm
   * \param vx Real world velocity in the x axis in cm/step
   * \param vy Real world velocity in the y axis in cm/step
   */
  Racket(int x, int y, int w, int h) : x{x}, y{y}, w{w}, h{h} {}

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
  // width in cm
  int w;
  // height in cm
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
      // TODO: Don't hardcode the width
      : racket_{0, 0, 5, static_cast<int>(height * .05)}, ball_{0, 0, 0, 0},
        width_{width}, height_{height}, proj_width_{viewport_x},
        proj_height_{viewport_y} {
    srand(time(nullptr));
  }

  void reset() {
    // Set random ball velocities
    ball_.vx = 0;
    while (ball_.vx == 0) {
      ball_.vx = -width_ * 0.01 * static_cast<double>(rand()) / RAND_MAX;
    }

    ball_.vy = -height_ * 0.01 * static_cast<double>(rand()) / RAND_MAX;

    // Position Racket and Ball in initial positions
    racket_.vx = 0;
    racket_.vy = 0;
    ball_.x = width_ - ball_.r - 1;
    ball_.y = ball_.r + (height_ - ball_.r * 2) * static_cast<double>(rand()) / RAND_MAX;
    racket_.x = racket_.w + 1;
    racket_.y = racket_.h + (height_ - racket_.h * 2) * static_cast<double>(rand()) / RAND_MAX;

    // Reset reward and completion flags
    reward_ = 0;
    done_ = false;
  }

  void step(Action action) {
    // Update ball and racket position
    ball_.update(1);

    // Make sure the ball stays in the court boundaries
    // We will want to add some randomness here, but not yet
    if (ball_.y + ball_.r >= height_ || ball_.y <= 0) {
      ball_.vy *= -1;
    }

    switch (action) {
    case Action::UP:
      racket_.vy = height_ * 0.008;
      break;
    case Action::DOWN:
      racket_.vy = -height_ * 0.008;
      break;
    default:
      racket_.vy = 0;
      break;
    }
    std::size_t y = racket_.y;
    racket_.update(1);
    // Make sure the racket statys in the court
    if (racket_.y + racket_.h >= height_ || racket_.y <= 0) {
      racket_.y = y;
    }

    // Check for collision AABB
    // Rightmost edge of Ball is always on the right with respect of the racket
    // leftmost edge
    if (racket_.x < ball_.x + ball_.r && racket_.x + racket_.w > ball_.x &&
        racket_.y < ball_.y + ball_.r && racket_.y + racket_.h > ball_.y) {
      // Collision!
      reward_ = 1;
      done_ = true;
    } else if (ball_.x <= 0) {
      // The reward will be the negative difference between the centers
      reward_ = -std::abs(ball_.y + ball_.r / 2 - racket_.y + racket_.h / 2) * yfactor;
      done_ = true;
    }
  }

  double reward() const { return reward_; }

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

  const Racket &racket() const { return racket_; }

  const Ball &ball() const { return ball_; }

  void render() const {
    auto &win_ = viz::Window::get();
    win_.clear();
    // Draw court with a linewidth of 2
    win_.add_rect(0, 0, width_, height_);
    win_.add_rect(1, 1, width_ - 2, height_ - 2);

    win_.fill_rect(ball_.x, ball_.y, ball_.r, ball_.r);
    win_.fill_rect(racket_.x, racket_.y, racket_.w, racket_.h);
    win_.present();
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
  double reward_ = 0.0;
  std::vector<double> state_ = std::vector<double>(8, 0);
  bool done_ = false;
  viz::Window *win_ = nullptr;
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

#define GAME_CAST(variable) auto &variable = *static_cast<Game *>(pong)

#define GEN_C_BINDING(name, fn)                 \
  void pong_##name(PongHdlr pong) {             \
      GAME_CAST(game);                          \
      game.fn();                                \
  }

#define GEN_C_BINDING_RET(name, fn, ret)            \
  ret pong_##name(PongHdlr pong) {             \
      GAME_CAST(game);                          \
      return game.fn();                                \
  }  

GEN_C_BINDING(reset, reset);
GEN_C_BINDING_RET(done,done, int);
GEN_C_BINDING(render,render);
GEN_C_BINDING_RET(reward, reward, double);
GEN_C_BINDING_RET(state, state().data, const double*);

}
