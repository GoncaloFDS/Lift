#pragma once

#include "core/utilities.h"
#include "events/event.h"
#include "window_data.h"
#include <GLFW/glfw3.h>
#include <functional>
#include <vector>

namespace vulkan {

using EventCallbackFn = std::function<void(Event &)>;

class Window final {
  public:
  explicit Window(const WindowData &config);
  ~Window();

  [[nodiscard]] const WindowData &config() const { return config_; }
  [[nodiscard]] GLFWwindow *handle() const { return handle_; }

  [[nodiscard]] static std::vector<const char *> getRequiredInstanceExtensions();
  [[nodiscard]] float contentScale() const;
  [[nodiscard]] static double time();
  [[nodiscard]] VkExtent2D framebufferSize() const;
  [[nodiscard]] VkExtent2D windowSize() const;

  [[nodiscard]] bool isMinimized() const;

  void close() const;
  static void waitForEvents();
  void poolEvents();

  void setEventCallbackFn(const EventCallbackFn &callback);

  private:
  WindowData config_;
  GLFWwindow *handle_{};
};

}// namespace vulkan
