#include "pch.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "glad/glad.h"
#include "scene.h"
#include "scene/cameras/camera.h"
#include "mesh.h"
#include "aabb.h"
#include "renderer/record.h"
#include "geometry_data.h"
#include "renderer/cuda_buffer.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include "tiny_gltf.h"
#include "core/profiler.h"
#include "cuda/vec_math.h"

template<typename T>
lift::BufferView<T> bufferViewFromGltf(const tinygltf::Model &model, lift::Scene *scene, const int32_t accessor_idx) {
    if (accessor_idx == -1)
        return lift::BufferView<T>();

    const auto &gltf_accessor = model.accessors[accessor_idx];
    const auto &gltf_buffer_view = model.bufferViews[gltf_accessor.bufferView];

    const int32_t elmt_byte_size =
        gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
        ? 2
        : gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT
          ? 4
          : gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
            ? 4
            : 0;
    LF_ASSERT(elmt_byte_size, "gltf accessor component type not supported");

    const CUdeviceptr buffer_base = scene->getBuffer(gltf_buffer_view.buffer);
    lift::BufferView<T> buffer_view;
    buffer_view.data = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride = static_cast<uint16_t>(gltf_buffer_view.byteStride);
    buffer_view.count = static_cast<uint32_t>(gltf_accessor.count);
    buffer_view.elmt_byte_size = static_cast<uint16_t>(elmt_byte_size);

    return buffer_view;
}

void contextLogCb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    LF_INFO("[OptiX Log] {0}", message);
}

void lift::Scene::addBuffer(const uint64_t buf_size, const void *data) {
    CUdeviceptr buffer = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer), buf_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(buffer),
                          data,
                          buf_size,
                          cudaMemcpyHostToDevice));
    buffers_.push_back(buffer);
}

void lift::Scene::addImage(const int32_t width, const int32_t height, const int32_t bits_per_component,
                           const int32_t num_components, const void *data) {
    Profiler profiler("addImage");
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
    res_desc.res.array.array = getImage(image_idx);

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

void lift::Scene::finalize() {
    createContext();
    buildMeshAccels();
    buildInstanceAccel();
    createPtxModule();
    createProgramGroups();
    createPipeline();
    createSbt();

    scene_aabb_.invalidate();
    for (const auto &mesh : meshes_)
        scene_aabb_.include(mesh->world_aabb);

    if (!cameras_.empty())
        cameras_.front().setLookAt(scene_aabb_.center());
}

void lift::Scene::cleanup() {
    // TODO
}

lift::Camera lift::Scene::getCamera() const {
    // TODO return set camera
    if (!cameras_.empty()) {
        LF_ERROR("Returning first camera");
        return cameras_.front();
    }
    LF_TRACE("Return default camera");
    Camera camera;
    camera.setFovy(45.0f);
    camera.setLookAt(scene_aabb_.center());
    camera.setEye(scene_aabb_.center() * vec3(0.0f, 0.0f, scene_aabb_.maxExtent()));
    return camera;

}

void lift::Scene::createContext() {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(nullptr));

    CUcontext cu_context = nullptr; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_context, &options, &context_));
}

void lift::Scene::buildMeshAccels() {
    Profiler profiler("BuildMeshAccels");
    // see explanation above
    constexpr double initial_compaction_ratio = 0.5;

    // It is assumed that trace is called later when the GASes are still in memory.
    // We know that the memory consumption at that time will at least be the compacted GASes + some CUDA stack space.
    // Add a "random" 250MB that we can use here, roughly matching CUDA stack space requirements.
    constexpr size_t additional_available_memory = 250 * 1024 * 1024;

    //////////////////////////////////////////////////////////////////////////

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    struct GasInfo {
        std::vector<OptixBuildInput> build_inputs;
        OptixAccelBufferSizes gas_buffer_sizes;
        std::shared_ptr<Mesh> mesh;
    };
    std::multimap<size_t, GasInfo> gases;
    size_t total_temp_output_size = 0;
    /*const*/
    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

    for (auto &mesh : meshes_) {
        const size_t num_sub_meshes = mesh->indices.size();
        std::vector<OptixBuildInput> build_inputs(num_sub_meshes);

        LF_ASSERT(mesh->positions.size() == num_sub_meshes &&
            mesh->normals.size() == num_sub_meshes &&
            mesh->tex_coords.size() == num_sub_meshes, "Mesh components size mismatch");

        for (size_t i = 0; i < num_sub_meshes; ++i) {
            OptixBuildInput &triangle_input = build_inputs[i];
            memset(&triangle_input, 0, sizeof(OptixBuildInput));
            triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes =
                mesh->positions[i].byte_stride ? mesh->positions[i].byte_stride : sizeof(float3),
                triangle_input.triangleArray.numVertices = mesh->positions[i].count;
            triangle_input.triangleArray.vertexBuffers = &(mesh->positions[i].data);
            triangle_input.triangleArray.indexFormat =
                mesh->indices[i].elmt_byte_size == 2
                ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3
                : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes =
                mesh->indices[i].byte_stride ? mesh->indices[i].byte_stride : mesh->indices[i].elmt_byte_size * 3;
            triangle_input.triangleArray.numIndexTriplets = mesh->indices[i].count / 3;
            triangle_input.triangleArray.indexBuffer = mesh->indices[i].data;
            triangle_input.triangleArray.flags = &triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accel_options, build_inputs.data(),
                                                 static_cast<unsigned int>( num_sub_meshes ), &gas_buffer_sizes));

        total_temp_output_size += gas_buffer_sizes.outputSizeInBytes;
        GasInfo g = {std::move(build_inputs), gas_buffer_sizes, mesh};
        gases.emplace(gas_buffer_sizes.outputSizeInBytes, g);
    }

    size_t total_temp_output_processed_size = 0;
    size_t used_compacted_output_size = 0;
    double compaction_ratio = initial_compaction_ratio;

    CudaBuffer<char> d_temp;
    CudaBuffer<char> d_temp_output;
    CudaBuffer<size_t> d_temp_compacted_sizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    while (!gases.empty()) {
        // The estimated total output size that we end up with when using compaction.
        // It defines the minimum peak memory consumption, but is unknown before actually building all GASes.
        // Working only within these memory constraints results in an actual peak memory consumption that is very close to the minimal peak memory consumption.
        auto remaining_estimated_total_output_size =
            static_cast<size_t>((total_temp_output_size - total_temp_output_processed_size) * compaction_ratio);
        auto available_mem_pool_size = remaining_estimated_total_output_size + additional_available_memory;
        // We need to fit the following things into availableMemPoolSize:
        // - temporary buffer for building a GAS (only during build, can be cleared before compaction)
        // - build output buffer of a GAS
        // - size (actual number) of a compacted GAS as output of a build
        // - compacted GAS

        size_t batch_nga_ses = 0;
        size_t batch_build_output_requirement = 0;
        size_t batch_build_max_temp_requirement = 0;
        size_t batch_build_compacted_requirement = 0;
        for (auto it = gases.rbegin(); it != gases.rend(); it++) {
            batch_build_output_requirement += it->second.gas_buffer_sizes.outputSizeInBytes;
            batch_build_compacted_requirement += (size_t) (it->second.gas_buffer_sizes.outputSizeInBytes *
                compaction_ratio);
            // roughly account for the storage of the compacted size, although that goes into a separate buffer
            batch_build_output_requirement += 8ull;
            // make sure that all further output pointers are 256 byte aligned
            batch_build_output_requirement = roundUp<size_t>(batch_build_output_requirement, 256ull);
            // temp buffer is shared for all builds in the batch
            batch_build_max_temp_requirement = std::max(batch_build_max_temp_requirement,
                                                        it->second.gas_buffer_sizes.tempSizeInBytes);
            batch_nga_ses++;
            if ((batch_build_output_requirement + batch_build_max_temp_requirement + batch_build_compacted_requirement)
                >
                    available_mem_pool_size)
                break;
        }

        // d_temp may still be available from a previous batch, but is freed later if it is "too big"
        d_temp.allocIfRequired(batch_build_max_temp_requirement);

        // trash existing buffer if it is more than 10% bigger than what we need
        // if it is roughly the same, we keep it
        if (d_temp_output.byteSize() > batch_build_output_requirement * 1.1)
            d_temp_output.free();
        d_temp_output.allocIfRequired(batch_build_output_requirement);

        // this buffer is assumed to be very small
        // trash d_temp_compactedSizes if it is at least 20MB in size and at least double the size than required for the next run
        if (d_temp_compacted_sizes.reservedCount() > batch_nga_ses * 2 && d_temp_compacted_sizes.byteSize() > 20 * 1024
            *
                1024)
            d_temp_compacted_sizes.free();
        d_temp_compacted_sizes.allocIfRequired(batch_nga_ses);

        // sum of build output size of GASes, excluding alignment
        size_t batch_temp_output_size = 0;
        // sum of size of compacted GASes
        size_t batch_compacted_size = 0;

        auto it = gases.rbegin();
        for (size_t i = 0, temp_output_alignment_offset = 0; i < batch_nga_ses; ++i) {
            emitProperty.result = d_temp_compacted_sizes.get(i);
            GasInfo &info = it->second;

            OPTIX_CHECK(optixAccelBuild(context_, nullptr, // CUDA stream
                                        &accel_options,
                                        info.build_inputs.data(),
                                        static_cast<unsigned int>( info.build_inputs.size()),
                                        d_temp.get(),
                                        d_temp.byteSize(),
                                        d_temp_output.get(temp_output_alignment_offset),
                                        info.gas_buffer_sizes.outputSizeInBytes,
                                        &info.mesh->gas_handle,
                                        &emitProperty, // emitted property list
                                        1 // num emitted properties
            ));

            temp_output_alignment_offset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
            it++;
        }

        // trash d_temp if it is at least 20MB in size
        if (d_temp.byteSize() > 20 * 1024 * 1024)
            d_temp.free();

        // download all compacted sizes to allocate final output buffers for these GASes
        std::vector<size_t> h_compacted_sizes(batch_nga_ses);
        d_temp_compacted_sizes.download(h_compacted_sizes.data());

        //////////////////////////////////////////////////////////////////////////
        // TODO:
        // Now we know the actual memory requirement of the compacted GASes.
        // Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
        bool can_compact = false;
        it = gases.rbegin();
        for (size_t i = 0; i < batch_nga_ses; ++i) {
            GasInfo &info = it->second;
            if (info.gas_buffer_sizes.outputSizeInBytes > h_compacted_sizes[i]) {
                can_compact = true;
                break;
            }
            it++;
        }

        if (can_compact) {
            //////////////////////////////////////////////////////////////////////////
            // "batch allocate" the compacted buffers
            it = gases.rbegin();
            for (size_t i = 0; i < batch_nga_ses; ++i) {
                GasInfo &info = it->second;
                batch_compacted_size += h_compacted_sizes[i];
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &info.mesh->d_gas_output ), h_compacted_sizes[i]));
                total_temp_output_processed_size += info.gas_buffer_sizes.outputSizeInBytes;
                it++;
            }

            it = gases.rbegin();
            for (size_t i = 0; i < batch_nga_ses; ++i) {
                GasInfo &info = it->second;
                OPTIX_CHECK(optixAccelCompact(context_, nullptr, info.mesh->gas_handle, info.mesh->d_gas_output,
                                              h_compacted_sizes[i], &info.mesh->gas_handle));
                it++;
            }
        } else {
            it = gases.rbegin();
            for (size_t i = 0, temp_output_alignment_offset = 0; i < batch_nga_ses; ++i) {
                GasInfo &info = it->second;
                info.mesh->d_gas_output = d_temp_output.get(temp_output_alignment_offset);
                batch_compacted_size += h_compacted_sizes[i];
                total_temp_output_processed_size += info.gas_buffer_sizes.outputSizeInBytes;

                temp_output_alignment_offset += roundUp<size_t>(info.gas_buffer_sizes.outputSizeInBytes, 256ull);
                it++;
            }
            d_temp_output.release();
        }

        used_compacted_output_size += batch_compacted_size;

        gases.erase(it.base(), gases.end());
    }
}

void lift::Scene::buildInstanceAccel(int ray_type_count) {
    Profiler profiler("BuildInstanceAccel");
    const size_t num_instances = meshes_.size();

    std::vector<OptixInstance> optix_instances(num_instances);

    unsigned int sbt_offset = 0;
    for (size_t i = 0; i < meshes_.size(); ++i) {
        auto mesh = meshes_[i];
        auto &optix_instance = optix_instances[i];
        memset(&optix_instance, 0, sizeof(OptixInstance));

        optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId = static_cast<unsigned int>(i);
        optix_instance.sbtOffset = sbt_offset;
        optix_instance.visibilityMask = 1;
        optix_instance.traversableHandle = mesh->gas_handle;

        memcpy(optix_instance.transform, value_ptr(mesh->transform), sizeof(float) * 12);
        // TODO ^ value_ptr(mesh->transform[0])

        sbt_offset += static_cast<unsigned int>(mesh->indices.size()) * ray_type_count;
        // one sbt record per GAS build input per RAY_TYPE
    }

    const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
    CUdeviceptr d_instances;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_instances ), instances_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>( d_instances ),
        optix_instances.data(),
        instances_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_,
        &accel_options,
        &instance_input,
        1, // num build inputs
        &ias_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>( &d_temp_buffer ),
        ias_buffer_sizes.tempSizeInBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>( &d_ias_output_buffer_ ),
        ias_buffer_sizes.outputSizeInBytes
    ));

    OPTIX_CHECK(optixAccelBuild(
        context_,
        nullptr, // CUDA stream
        &accel_options,
        &instance_input,
        1, // num build inputs
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_ias_output_buffer_,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle_,
        nullptr, // emitted property list
        0 // num emitted properties
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>( d_temp_buffer )));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>( d_instances )));
}

void lift::Scene::loadFromFile(const std::string &file_name) {
    Profiler profiler("Load Scene");
    LF_INFO("Loading Scene");
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    bool ret;
    {
        Profiler profiler_1("LoadACIIFromFile");
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, file_name);
    }
    if (!warn.empty())
        LF_ERROR("glTF Warning: {0}", warn);

    LF_ASSERT(ret, "Failed to load GLTF Scene {0}: {1}", file_name, err);

    //
    // Process buffer data first -- buffer views will reference this list
    //
    for (const auto &gltf_buffer : model.buffers) {
        const uint64_t buf_size = gltf_buffer.data.size();
        addBuffer(buf_size, gltf_buffer.data.data());
    }

    //
    // Images -- just load all up front for simplicity
    //
    for (const auto &gltf_image : model.images) {
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
    for (const auto &gltf_texture : model.textures) {
        if (gltf_texture.sampler == -1) {
            addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source);
            continue;
        }

        const auto &gltf_sampler = model.samplers[gltf_texture.sampler];

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
    for (auto &gltf_material : model.materials) {
        MaterialData mtl;

        const auto base_color_it = gltf_material.values.find("baseColorFactor");
        if (base_color_it != gltf_material.values.end()) {
            const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
            mtl.base_color = make_float4(float(c[0]), float(c[1]), float(c[2]), float(c[3]));
        }
        const auto base_color_t_it = gltf_material.values.find("baseColorTexture");
        if (base_color_t_it != gltf_material.values.end()) {
            mtl.base_color_tex = getSampler(base_color_t_it->second.TextureIndex());
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
            mtl.metallic_roughness_tex = getSampler(metallic_roughness_it->second.TextureIndex());
        }
        const auto normal_it = gltf_material.additionalValues.find("normalTexture");
        if (normal_it != gltf_material.additionalValues.end()) {
            mtl.normal_tex = getSampler(normal_it->second.TextureIndex());
        }

        addMaterial(mtl);
    }

    //
    // Process nodes
    //
    std::vector<int32_t> root_nodes(model.nodes.size(), 1);
    for (auto &gltf_node : model.nodes)
        for (int32_t child : gltf_node.children)
            root_nodes[child] = 0;

    for (size_t i = 0; i < root_nodes.size(); ++i) {
        if (!root_nodes[i])
            continue;
        auto &gltf_node = model.nodes[i];

        processGltfNode(model, gltf_node, mat4(1.0f));
    }

}

void lift::Scene::processGltfNode(const tinygltf::Model &model, const tinygltf::Node &gltf_node,
                                  const mat4 parent_matrix) {
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
                        : transpose(make_mat4(reinterpret_cast<float *>(gltf_matrix.data())));

    const mat4 node_xform = parent_matrix * matrix * translation * rotation * scale;

    if (gltf_node.camera != -1) {
        const auto &gltf_camera = model.cameras[gltf_node.camera];
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

        Camera camera;
        camera.setFovy(yfov);
        camera.setAspectRatio(static_cast<float>(gltf_camera.perspective.aspectRatio));
        camera.setEye(eye);
        camera.setUp(up);
        addCamera(camera);
    } else if (gltf_node.mesh != -1) {
        const auto &gltf_mesh = model.meshes[gltf_node.mesh];
        for (auto &gltf_primitive : gltf_mesh.primitives) {
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

            const auto &pos_gltf_accessor = model.accessors[pos_accessor_idx];
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

void lift::Scene::createPtxModule() {

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    pipeline_compile_options_ = {};
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options_.numPayloadValues = k_NumPayloadValues;
    pipeline_compile_options_.numAttributeValues = 2; // TODO
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";

    const auto ptx = Util::getPtxString("res/ptx/device_programs.ptx");

    ptx_module_ = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
        context_,
        &module_compile_options,
        &pipeline_compile_options_,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &ptx_module_
    ));
}

void lift::Scene::createProgramGroups() {
    OptixProgramGroupOptions program_group_options = {};

    char log[2048];
    size_t sizeof_log = sizeof(log);

    //
    // Ray generation
    //
    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module_;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__render_frame";

        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &raygen_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group_
        )
        );
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module_;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &miss_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &radiance_miss_group_
        )
        );

        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr; // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &miss_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &occlusion_miss_group_
        )
        );
    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = ptx_module_;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &hit_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &radiance_hit_group_
        )
        );

        memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = nullptr;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            context_,
            &hit_prog_group_desc,
            1, // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &occlusion_hit_group_
        )
        );
    }
}

void lift::Scene::createPipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_prog_group_,
        radiance_miss_group_,
        occlusion_miss_group_,
        radiance_hit_group_,
        occlusion_hit_group_
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = k_MaxTraceDepth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    pipeline_link_options.overrideUsesMotionBlur = false;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context_,
        &pipeline_compile_options_,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline_
    ));
}

void lift::Scene::createSbt() {
    {
        const size_t raygen_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &sbt_.raygenRecord ), raygen_record_size));

        EmptyRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>( sbt_.raygenRecord ),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice
        ));
    }

    {
        const size_t miss_record_size = sizeof(EmptyRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>( &sbt_.missRecordBase ),
            miss_record_size * RAY_TYPE_COUNT
        ));

        EmptyRecord ms_sbt[RAY_TYPE_COUNT];
        OPTIX_CHECK(optixSbtRecordPackHeader(radiance_miss_group_, &ms_sbt[0]));
        OPTIX_CHECK(optixSbtRecordPackHeader(occlusion_miss_group_, &ms_sbt[1]));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>( sbt_.missRecordBase ),
            ms_sbt,
            miss_record_size * RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ));
        sbt_.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
        sbt_.missRecordCount = RAY_TYPE_COUNT;
    }

    {
        std::vector<HitGroupRecord> hitgroup_records;
        for (const auto &mesh : meshes_) {
            for (size_t i = 0; i < mesh->material_idx.size(); ++i) {
                HitGroupRecord rec = {};
                OPTIX_CHECK(optixSbtRecordPackHeader(radiance_hit_group_, &rec));
                rec.data.geometry_data.type = GeometryData::TRIANGLE_MESH;
                rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
                rec.data.geometry_data.triangle_mesh.normals = mesh->normals[i];
                rec.data.geometry_data.triangle_mesh.tex_coords = mesh->tex_coords[i];
                rec.data.geometry_data.triangle_mesh.indices = mesh->indices[i];

                const int32_t mat_idx = mesh->material_idx[i];
                if (mat_idx >= 0)
                    rec.data.material_data = materials_[mat_idx];
                else
                    rec.data.material_data = MaterialData();
                hitgroup_records.push_back(rec);

                OPTIX_CHECK(optixSbtRecordPackHeader(occlusion_hit_group_, &rec));
                hitgroup_records.push_back(rec);
            }
        }

        const size_t hitgroup_record_size = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>( &sbt_.hitgroupRecordBase ),
            hitgroup_record_size * hitgroup_records.size()
        ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>( sbt_.hitgroupRecordBase ),
            hitgroup_records.data(),
            hitgroup_record_size * hitgroup_records.size(),
            cudaMemcpyHostToDevice
        ));

        sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
        sbt_.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
    }

}
