#include "pch.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <renderer/cuda_buffer.h>
#include "glad/glad.h"
#include "scene.h"
#include "scene/cameras/camera.h"
#include "mesh.h"
#include "aabb.h"
#include "renderer/record.h"
#include "cuda/geometry_data.h"
#include "renderer/cuda_output_buffer.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include "tiny_gltf.h"
#include "core/profiler.h"
#include "cuda/vec_math.h"


template<typename T>
auto bufferViewFromGltf(const tinygltf::Model& model,
                        lift::Scene* scene,
                        const int32_t accessor_idx) -> lift::BufferView<T> {
    if (accessor_idx == -1)
        return lift::BufferView<T>();

    const auto& gltf_accessor = model.accessors[accessor_idx];
    const auto& gltf_buffer_view = model.bufferViews[gltf_accessor.bufferView];

    int32_t elmt_byte_size;
    switch (gltf_accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:elmt_byte_size = 2;
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        case TINYGLTF_COMPONENT_TYPE_FLOAT:elmt_byte_size = 4;
            break;
        default:elmt_byte_size = 0;
    }
    LF_ASSERT(elmt_byte_size, "gltf accessor component type not supported");
    const CUdeviceptr buffer_base = scene->buffer(gltf_buffer_view.buffer);
    lift::BufferView<T> buffer_view;
    buffer_view.data = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride = static_cast<uint16_t>(gltf_buffer_view.byteStride);
    buffer_view.count = static_cast<uint32_t>(gltf_accessor.count);
    buffer_view.elmt_byte_size = static_cast<uint16_t>(elmt_byte_size);

    return buffer_view;
}

void lift::Scene::addBuffer(const uint64_t buf_size, const void* data) {
    CUdeviceptr buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer), buf_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer),
                          data,
                          buf_size,
                          cudaMemcpyHostToDevice));
    buffers_.push_back(buffer);
}

void lift::Scene::addImage(const int32_t width, const int32_t height, const int32_t bits_per_component,
                           const int32_t num_components, const void* data) {
    // Allocate CUDA array in device memory
    int32_t pitch = 0;
    cudaChannelFormatDesc channel_desc{};
    if (bits_per_component == 8) {
        pitch = width * num_components * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    } else if (bits_per_component == 16) {
        pitch = width * num_components * sizeof(uint16_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    } else {
        LF_ASSERT(false, "Unsupported bits/component in glTF image");
    }

    cudaArray_t cuda_array = nullptr;
    CUDA_CHECK(cudaMallocArray(
        &cuda_array,
        &channel_desc,
        width,
        height
    ));
    CUDA_CHECK(cudaMemcpy2DToArray(
        cuda_array,
        0, // X offset
        0, // Y offset
        data,
        pitch,
        pitch,
        height,
        cudaMemcpyHostToDevice
    ));
    images_.push_back(cuda_array);
}

void lift::Scene::addSampler(cudaTextureAddressMode address_s, cudaTextureAddressMode address_t,
                             cudaTextureFilterMode filter_mode, const int32_t image_idx) {
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = image(image_idx);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = address_s == GL_CLAMP_TO_EDGE
                              ? cudaAddressModeClamp
                              : address_s == GL_MIRRORED_REPEAT
                                ? cudaAddressModeMirror
                                : cudaAddressModeWrap;
    tex_desc.addressMode[1] = address_t == GL_CLAMP_TO_EDGE
                              ? cudaAddressModeClamp
                              : address_t == GL_MIRRORED_REPEAT
                                ? cudaAddressModeMirror
                                : cudaAddressModeWrap;
    tex_desc.filterMode = filter_mode == GL_NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0; // TODO: glTF assumes sRGB for base_color -- handle in shader

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    samplers_.push_back(cuda_tex);
}

void lift::Scene::calculateAabb() {
    scene_aabb_.invalidate();
    for (const auto& mesh : meshes_)
        scene_aabb_.include(mesh->world_aabb);

}

void lift::Scene::cleanup() {
    // TODO
}

auto lift::Scene::camera() -> std::shared_ptr<lift::Camera> {
    if (cameras_.empty()) {
        auto camera = std::make_shared<Camera>();
        camera->setFovy(45.0f);
        camera->setLookAt(scene_aabb_.center());
        camera->setEye(scene_aabb_.center() + vec3(1.0f, 1.0f, 1.0f));
        addCamera(camera);
    }
    return cameras_.front();
}

void lift::Scene::loadFromFile(const std::string& file_name) {
    Profiler profiler("Load Scene");
    LF_INFO("Loading Scene {0}", file_name);
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, file_name);
    if (!warn.empty())
        LF_ERROR("glTF Warning: {0}", warn);

    LF_ASSERT(ret, "Failed to load GLTF Scene {0}: {1}", file_name, err);

    //
    // Process buffer data first -- buffer views will reference this list
    //
    for (const auto& gltf_buffer : model.buffers) {
        const uint64_t buf_size = gltf_buffer.data.size();
        addBuffer(buf_size, gltf_buffer.data.data());
    }

    //
    // Images -- just load all up front for simplicity
    //
    for (const auto& gltf_image : model.images) {
        assert(gltf_image.component == 4);
        assert(gltf_image.bits == 8 || gltf_image.bits == 16);

        addImage(
            gltf_image.width,
            gltf_image.height,
            gltf_image.bits,
            gltf_image.component,
            gltf_image.image.data()
        );
    }

    //
    // Textures -- refer to previously loaded images
    //
    for (const auto& gltf_texture : model.textures) {
        if (gltf_texture.sampler == -1) {
            addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source);
            continue;
        }

        const auto& gltf_sampler = model.samplers[gltf_texture.sampler];

        const cudaTextureAddressMode address_s = gltf_sampler.wrapS == GL_CLAMP_TO_EDGE
                                                 ? cudaAddressModeClamp
                                                 : gltf_sampler.wrapS == GL_MIRRORED_REPEAT
                                                   ? cudaAddressModeMirror
                                                   : cudaAddressModeWrap;
        const cudaTextureAddressMode address_t = gltf_sampler.wrapT == GL_CLAMP_TO_EDGE
                                                 ? cudaAddressModeClamp
                                                 : gltf_sampler.wrapT == GL_MIRRORED_REPEAT
                                                   ? cudaAddressModeMirror
                                                   : cudaAddressModeWrap;
        const cudaTextureFilterMode filter = gltf_sampler.minFilter == GL_NEAREST
                                             ? cudaFilterModePoint
                                             : cudaFilterModeLinear;
        addSampler(address_s, address_t, filter, gltf_texture.source);
    }

    //
    // Materials
    //
    for (auto& gltf_material : model.materials) {
        MaterialData mtl;

        const auto base_color_it = gltf_material.values.find("baseColorFactor");
        if (base_color_it != gltf_material.values.end()) {
            const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
            mtl.base_color = make_float4(float(c[0]), float(c[1]), float(c[2]), float(c[3]));
        }
        const auto base_color_t_it = gltf_material.values.find("baseColorTexture");
        if (base_color_t_it != gltf_material.values.end()) {
            mtl.base_color_tex = sampler(base_color_t_it->second.TextureIndex());
        }
        const auto roughness_it = gltf_material.values.find("roughnessFactor");
        if (roughness_it != gltf_material.values.end()) {
            mtl.roughness = static_cast<float>(roughness_it->second.Factor());
        }
        const auto metallic_it = gltf_material.values.find("metallicFactor");
        if (metallic_it != gltf_material.values.end()) {
            mtl.metallic = static_cast<float>(metallic_it->second.Factor());
        }
        const auto metallic_roughness_it = gltf_material.values.find("metallicRoughnessTexture");
        if (metallic_roughness_it != gltf_material.values.end()) {
            mtl.metallic_roughness_tex = sampler(metallic_roughness_it->second.TextureIndex());
        }
        const auto normal_it = gltf_material.additionalValues.find("normalTexture");
        if (normal_it != gltf_material.additionalValues.end()) {
            mtl.normal_tex = sampler(normal_it->second.TextureIndex());
        }

        addMaterial(mtl);
    }

    //
    // Process nodes
    //
    std::vector<int32_t> root_nodes(model.nodes.size(), 1);
    for (auto& gltf_node : model.nodes)
        for (int32_t child : gltf_node.children)
            root_nodes[child] = 0;

    for (size_t i = 0; i < root_nodes.size(); ++i) {
        if (!root_nodes[i])
            continue;
        auto& gltf_node = model.nodes[i];

        processGltfNode(model, gltf_node, mat4(1.0f));
    }

}

void lift::Scene::processGltfNode(const tinygltf::Model& model, const tinygltf::Node& gltf_node,
                                  const mat4& parent_matrix) {
    const mat4 translation = gltf_node.translation.empty()
                             ? mat4(1.0f)
                             : translate(mat4(1.0f), vec3(
            gltf_node.translation[0],
            gltf_node.translation[1],
            gltf_node.translation[2]
        ));

    const mat4 rotation = gltf_node.rotation.empty()
                          ? mat4(1.0f)
                          : toMat4(quat(
            static_cast<float>(gltf_node.rotation[3]),
            static_cast<float>(gltf_node.rotation[0]),
            static_cast<float>(gltf_node.rotation[1]),
            static_cast<float>(gltf_node.rotation[2])
        ));

    const mat4 scale = gltf_node.scale.empty()
                       ? mat4(1.0f)
                       : glm::scale(mat4(1.0f), vec3(
            gltf_node.scale[0],
            gltf_node.scale[1],
            gltf_node.scale[2]
        ));

    std::vector<float> gltf_matrix;
    for (double x : gltf_node.matrix)
        gltf_matrix.push_back(static_cast<float>(x));
    const mat4 matrix = gltf_node.matrix.empty()
                        ? mat4(1.0f)
                        : (make_mat4(reinterpret_cast<float*>(gltf_matrix.data())));

    mat4 node_xform = parent_matrix * matrix * translation * rotation * scale;

    if (gltf_node.camera != -1) {
        const auto& gltf_camera = model.cameras[gltf_node.camera];
        if (gltf_camera.type != "perspective") {
            return;
        }

        const vec3 eye = vec3(node_xform * vec4(0.0f, 0.0f, 0.0f, 1.0f));
        const vec3 up = vec3(node_xform * vec4(0.0f, 1.0f, 0.0f, 0.0f));
        const float yfov = static_cast<float>(gltf_camera.perspective.yfov) * 180.0f / pi<float>();

        std::cerr << "\teye   : " << eye.x << ", " << eye.y << ", " << eye.z << std::endl;
        std::cerr << "\tup    : " << up.x << ", " << up.y << ", " << up.z << std::endl;
        std::cerr << "\tfov   : " << yfov << std::endl;
        std::cerr << "\taspect: " << gltf_camera.perspective.aspectRatio << std::endl;

        auto camera = std::make_shared<Camera>();
        camera->setFovy(yfov);
        camera->setAspectRatio(static_cast<float>(gltf_camera.perspective.aspectRatio));
        camera->setEye(eye);
        camera->setUp(up);
        addCamera(camera);
    } else if (gltf_node.mesh != -1) {
        const auto& gltf_mesh = model.meshes[gltf_node.mesh];
        for (auto& gltf_primitive : gltf_mesh.primitives) {
            if (gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES) // Ignore non-triangle meshes
            {
                std::cerr << "\tNon-triangle primitive: skipping\n";
                continue;
            }

            auto mesh = std::make_shared<Mesh>();
            addMesh(mesh);

            mesh->name = gltf_mesh.name;
            mesh->indices.push_back(bufferViewFromGltf<uint32_t>(model, this, gltf_primitive.indices));
            mesh->material_idx.push_back(gltf_primitive.material);
            mesh->transform = node_xform;

            assert(gltf_primitive.attributes.find("POSITION") != gltf_primitive.attributes.end());
            const int32_t pos_accessor_idx = gltf_primitive.attributes.at("POSITION");
            mesh->positions.push_back(bufferViewFromGltf<float3>(model, this, pos_accessor_idx));

            const auto& pos_gltf_accessor = model.accessors[pos_accessor_idx];
            mesh->object_aabb = Aabb(
                vec3(
                    pos_gltf_accessor.minValues[0],
                    pos_gltf_accessor.minValues[1],
                    pos_gltf_accessor.minValues[2]
                ),
                vec3(
                    pos_gltf_accessor.maxValues[0],
                    pos_gltf_accessor.maxValues[1],
                    pos_gltf_accessor.maxValues[2]
                ));
            mesh->world_aabb = mesh->object_aabb;
            mesh->world_aabb.transform(node_xform);

            auto normal_accessor_iter = gltf_primitive.attributes.find("NORMAL");
            if (normal_accessor_iter != gltf_primitive.attributes.end()) {
                mesh->normals.push_back(bufferViewFromGltf<float3>(model, this, normal_accessor_iter->second));
            } else {
                mesh->normals.push_back(bufferViewFromGltf<float3>(model, this, -1));
            }

            auto texcoord_accessor_iter = gltf_primitive.attributes.find("TEXCOORD_0");
            if (texcoord_accessor_iter != gltf_primitive.attributes.end()) {
                mesh->tex_coords.push_back(bufferViewFromGltf<float2>(model, this, texcoord_accessor_iter->second));
            } else {
                mesh->tex_coords.push_back(bufferViewFromGltf<float2>(model, this, -1));
            }
        }
    } else if (!gltf_node.children.empty()) {
        for (int32_t child : gltf_node.children) {
            processGltfNode(model, model.nodes[child], node_xform);
        }
    }
}


