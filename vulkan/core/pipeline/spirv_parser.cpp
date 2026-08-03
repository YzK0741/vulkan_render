//
// Created by 小叶 on 2026/8/2.
//
module;

#include <filesystem>
#include <fstream>
#include <optional>
#include "../../../spirv-reflect/spirv_reflect.h"

module vulkan.pipeline.spirv_parser;
import utility;

namespace {

    std::vector<vulkan::pipeline::located_variable> parse_variables(std::vector<SpvReflectInterfaceVariable *> const& variables) {

    }

}


std::optional<vulkan::pipeline::shader> vulkan::pipeline::load_shader(std::string_view filename) {
    const std::filesystem::path path(filename);

    if (!std::filesystem::is_regular_file(path) || filename.ends_with(".spv")) {
        return std::nullopt;
    }

    uint64_t file_size = std::filesystem::file_size(path);
    std::ifstream file(path);
    std::vector<unsigned char> code(file_size);

    file.read(reinterpret_cast<char*>(code.data()), static_cast<long long>(code.size()));

    SpvReflectShaderModule module = {};
    spvReflectCreateShaderModule(file_size, code.data(), &module);

    const SpvReflectEntryPoint* entry_point = spvReflectGetEntryPoint(&module, "main");

    if (entry_point == nullptr) {
        return std::nullopt;
    }

    uint32_t count;

    spvReflectEnumerateInputVariables(&module, &count, nullptr);
    std::vector<SpvReflectInterfaceVariable *> input_variables;
    spvReflectEnumerateInputVariables(&module, &count, input_variables.data());



    spvReflectEnumerateOutputVariables(&module, &count, nullptr);
    std::vector<SpvReflectInterfaceVariable *> output_variables;
    spvReflectEnumerateOutputVariables(&module, &count, output_variables.data());

    spvReflectEnumeratePushConstantBlocks(&module, &count, nullptr);
    std::vector<SpvReflectBlockVariable *> push_constants;
    spvReflectEnumeratePushConstantBlocks(&module, &count, push_constants.data());

    spvReflectEnumerateDescriptorBindings(&module, &count, nullptr);
    std::vector<SpvReflectDescriptorBinding *> bindings(count);
    spvReflectEnumerateDescriptorBindings(&module, &count, bindings.data());



    return std::nullopt;
}
