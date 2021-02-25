#pragma once

#include "SDL_rect.h"
#include "SDL_render.h"
#include "SDL_surface.h"
#include "SDL_video.h"
#include <SDL.h>

namespace viz {

class Window {
public:
  // Singleton getter
  static Window &get() {
    static Window w{800, 600};
    return w;
  }

  // Release SDL resources
  ~Window() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
  }

  // Clear the Window with White
  void clear() {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
  }

  void add_rect(int x, int y, int w, int h) {
    auto rect = SDL_Rect();
    rect.x = x;
    rect.y = y;
    rect.w = w;
    rect.h = h;
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);    
    SDL_RenderDrawRect(renderer, &rect);
  }

  void fill_rect(int x, int y, int w, int h) {
    auto rect = SDL_Rect();
    rect.x = x;
    rect.y = y;
    rect.w = w;
    rect.h = h;
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);    
    SDL_RenderFillRect(renderer, &rect);
  }  

  void present() {
    SDL_RenderPresent(renderer);    
  }

private:
  SDL_Window *win;
  SDL_Renderer *renderer;

  Window(int, int) {
    SDL_Init(SDL_INIT_VIDEO);
    win = SDL_CreateWindow("This is my window", 200, 200, 800, 600,
                           SDL_WINDOW_OPENGL);
    renderer = SDL_CreateRenderer(
        win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  }

  // Protect the singleton
  Window(const Window &) = delete;
  Window(Window &&) = delete;
  Window &operator=(const Window &) = delete;
  Window &operator=(const Window &&) = delete;
};

} // namespace viz
