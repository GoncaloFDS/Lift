
#include "options.h"
#include "core.h"
#include "scene_list.h"
#include <boost/program_options.hpp>

using namespace boost::program_options;

Options::Options(const int argc, const char* argv[]) {
    options_description benchmark_options("Benchmark options");
    benchmark_options.add_options()("next-scenes",
                                    bool_switch(&benchmark_next_scenes)->default_value(false),
                                    "Load the next scene once the sample or time limit is reached.")(
        "max-time",
        value<uint32_t>(&benchmark_max_time)->default_value(60),
        "The benchmark time limit per scene (in seconds).");

    options_description renderer("Renderer options");
    renderer.add_options()("samples",
                           value<uint32_t>(&samples)->default_value(4),
                           "Set the number of ray samples per pixel.")("bounces",
                                                                       value<uint32_t>(&bounces)->default_value(8),
                                                                       "Set the maximum number of bounces per ray.")(
        "max-samples",
        value<uint32_t>(&max_samples)->default_value(64 * 1024),
        "Set the maximum number of accumulated ray samples per pixel.");

    options_description scene("Scene options");
    scene.add_options()("scene", value<uint32_t>(&scene_index)->default_value(0), "Set the scene to start with.");
    options_description algorithm("Algorithm options");
    algorithm.add_options()("algorithm", value<uint32_t>(&algorithm_index)->default_value(0), "Set the algorithm to start with.");

    options_description window("Window options");
    window.add_options()("width", value<uint32_t>(&width)->default_value(800), "Set framebuffer width.")(
        "height",
        value<uint32_t>(&height)->default_value(600),
        "Set framebuffer height.")("fullscreen",
                                   bool_switch(&fullscreen)->default_value(false),
                                   "Toggle fullscreen vs windowed (default: windowed).")(
        "vsync",
        bool_switch(&vSync)->default_value(false),
        "Toggle vsync (default: vsync off).");

    options_description desc("Application options");
    desc.add_options()("help", "Display help message.")("benchmark",
                                                        bool_switch(&benchmark)->default_value(false),
                                                        "Run the application in benchmark mode.");

    desc.add(benchmark);
    desc.add(renderer);
    desc.add(scene);
    desc.add(algorithm);
    desc.add(window);

    const positional_options_description positional;
    variables_map vm;
    store(command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
    notify(vm);

    LF_ASSERT(scene_index < SceneList::all_scenes.size(), "Scene index is too large");
}
