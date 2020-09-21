#include "options.h"
#include "algorithm_list.h"
#include "core.h"
#include "scene_list.h"
#include <boost/program_options.hpp>

using namespace boost::program_options;

Options::Options(const int argc, const char* argv[]) {
    bool help;
    options_description desc("");
    desc.add_options()
        ("help,h", bool_switch(&help)->default_value(false), "display cli arguments");

    options_description renderer("renderer");
    renderer.add_options()
        ("samples,n", value<uint32_t>(&samples)->default_value(1), "number of rays per pixel")
        ("bounces,b", value<uint32_t>(&bounces)->default_value(32), "maximum path length")
        ("max-samples", value<uint32_t>(&max_samples)->default_value(2048), "accumulation target");

    options_description algorithm("algorithm");
    algorithm.add_options()
        ("algorithm,a",value<uint32_t>(&algorithm_index)->default_value(0),"algorithm index");

    options_description scene("scene");
    scene.add_options()
        ("scene,s", value<uint32_t>(&scene_index)->default_value(0), "scene index");

    options_description window("window");
    window.add_options()
        ("width", value<uint32_t>(&width)->default_value(1920), "window width")
        ("height", value<uint32_t>(&height)->default_value(1080), "window height")
        ("fullscreen", bool_switch(&fullscreen)->default_value(false), "fullscreen or windowed");

    options_description benchmark_options("benchmark");
    benchmark_options.add_options()
        ("benchmark", bool_switch(&benchmark)->default_value(false),"benchmark mode")
        ("all", bool_switch(&benchmark_next_scenes)->default_value(false), "benchmark every scene");

    desc.add(renderer);
    desc.add(algorithm);
    desc.add(scene);
    desc.add(window);
    desc.add(benchmark_options);

    const positional_options_description positional;
    variables_map vm;
    store(command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
    notify(vm);

    if(help) {
        LF_INFO("\n{}", desc);
        exit(EXIT_SUCCESS);
    }

    if (scene_index >= SceneList::all_scenes.size()) {
        LF_ERROR("scene index is too large");
        exit(EXIT_FAILURE);
    }
    if (algorithm_index >= AlgorithmList::all_algorithms.size()) {
        LF_ERROR("algorithm index is too large");
        exit(EXIT_FAILURE);
    }
}
