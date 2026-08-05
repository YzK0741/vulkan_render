//
// Created by 小叶 on 2026/8/2.
//
module;

#include <algorithm>
#include <expected>
#include <span>
#include <vulkan/vulkan.h>
#include "../../../third_party/spirv-reflect/spirv_reflect.h"

module vulkan.pipeline.spirv_parser;
import utility;

namespace {
  vulkan::pipeline::interface_variable_info reflect_var_to_info(const SpvReflectInterfaceVariable* p_var) {
    vulkan::pipeline::interface_variable_info info = {};
    info.location = p_var->location;
    info.component = p_var->component;
    info.format = static_cast<VkFormat>(p_var->format);
    info.name = p_var->name ? p_var->name : "unnamed";
    info.is_builtin = (p_var->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN) != 0;
    return info;
  }

  void sort_interface_variables(std::vector<vulkan::pipeline::interface_variable_info>& vars) {
      std::ranges::sort(vars,
      [](const vulkan::pipeline::interface_variable_info& a, const vulkan::pipeline::interface_variable_info& b) {
        if (a.location != b.location) return a.location < b.location;
        return a.component < b.component;
      });
  }
}

uint32_t vulkan::pipeline::format_size(const VkFormat format) {
    uint32_t result = 0;
    switch (format) {
      case VK_FORMAT_UNDEFINED:
        result = 0;
        break;
        case VK_FORMAT_R4G4_UNORM_PACK8:
          result = 1;
          break;
        case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
        case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
        case VK_FORMAT_R5G6B5_UNORM_PACK16:
        case VK_FORMAT_B5G6R5_UNORM_PACK16:
        case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
        case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
        case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
          result = 2;
          break;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8_SNORM:
        case VK_FORMAT_R8_USCALED:
        case VK_FORMAT_R8_SSCALED:
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8_SINT:
        case VK_FORMAT_R8_SRGB:
          result = 1;
          break;
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8_SNORM:
        case VK_FORMAT_R8G8_USCALED:
        case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8_SRGB:
          result = 2;
          break;
        case VK_FORMAT_R8G8B8_UNORM:
        case VK_FORMAT_R8G8B8_SNORM:
        case VK_FORMAT_R8G8B8_USCALED:
        case VK_FORMAT_R8G8B8_SSCALED:
        case VK_FORMAT_R8G8B8_UINT:
        case VK_FORMAT_R8G8B8_SINT:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_UNORM:
        case VK_FORMAT_B8G8R8_SNORM:
        case VK_FORMAT_B8G8R8_USCALED:
        case VK_FORMAT_B8G8R8_SSCALED:
        case VK_FORMAT_B8G8R8_UINT:
        case VK_FORMAT_B8G8R8_SINT:
        case VK_FORMAT_B8G8R8_SRGB:
          result = 3;
          break;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SNORM:
        case VK_FORMAT_R8G8B8A8_USCALED:
        case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R8G8B8A8_SINT:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_B8G8R8A8_USCALED:
        case VK_FORMAT_B8G8R8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_UINT:
        case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_USCALED_PACK32:
        case VK_FORMAT_A8B8G8R8_SSCALED_PACK32:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32:
        case VK_FORMAT_A8B8G8R8_SINT_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_SNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_USCALED_PACK32:
        case VK_FORMAT_A2R10G10B10_SSCALED_PACK32:
        case VK_FORMAT_A2R10G10B10_UINT_PACK32:
        case VK_FORMAT_A2R10G10B10_SINT_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
        case VK_FORMAT_A2B10G10R10_USCALED_PACK32:
        case VK_FORMAT_A2B10G10R10_SSCALED_PACK32:
        case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        case VK_FORMAT_A2B10G10R10_SINT_PACK32:
          result = 4;
          break;
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16_USCALED:
        case VK_FORMAT_R16_SSCALED:
        case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16_SINT:
        case VK_FORMAT_R16_SFLOAT:
          result = 2;
          break;
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
        case VK_FORMAT_R16G16_USCALED:
        case VK_FORMAT_R16G16_SSCALED:
        case VK_FORMAT_R16G16_UINT:
        case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_SFLOAT:
          result = 4;
          break;
        case VK_FORMAT_R16G16B16_UNORM:
        case VK_FORMAT_R16G16B16_SNORM:
        case VK_FORMAT_R16G16B16_USCALED:
        case VK_FORMAT_R16G16B16_SSCALED:
        case VK_FORMAT_R16G16B16_UINT:
        case VK_FORMAT_R16G16B16_SINT:
        case VK_FORMAT_R16G16B16_SFLOAT:
          result = 6;
          break;
        case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_R16G16B16A16_SNORM:
        case VK_FORMAT_R16G16B16A16_USCALED:
        case VK_FORMAT_R16G16B16A16_SSCALED:
        case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SINT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
          result = 8;
          break;
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32_SFLOAT:
          result = 4;
          break;
        case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT:
        case VK_FORMAT_R32G32_SFLOAT:
          result = 8;
          break;
        case VK_FORMAT_R32G32B32_UINT:
        case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32_SFLOAT:
          result = 12;
          break;
        case VK_FORMAT_R32G32B32A32_UINT:
        case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
          result = 16;
          break;
        case VK_FORMAT_R64_UINT:
          result = 8;
          break;
        case VK_FORMAT_R64_SINT:
        case VK_FORMAT_R64_SFLOAT:
        case VK_FORMAT_R64G64_UINT:
        case VK_FORMAT_R64G64_SINT:
        case VK_FORMAT_R64G64_SFLOAT:
          result = 16;
          break;
        case VK_FORMAT_R64G64B64_UINT:
        case VK_FORMAT_R64G64B64_SINT:
        case VK_FORMAT_R64G64B64_SFLOAT:
          result = 24;
          break;
        case VK_FORMAT_R64G64B64A64_UINT:
        case VK_FORMAT_R64G64B64A64_SINT:
        case VK_FORMAT_R64G64B64A64_SFLOAT:
          result = 32;
          break;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
          result = 4;
          break;

        default:
          break;
      }
      return result;
}


std::expected<vulkan::pipeline::shader_stage_interface, std::string_view>
vulkan::pipeline::parse_shader_stage_interface(
    const std::span<unsigned char> spirv_code,
    const VkShaderStageFlagBits stage){
     using fail = std::unexpected<std::string_view>;

      shader_stage_interface out_interface;
      out_interface.stage = stage;
      out_interface.inputs.clear();
      out_interface.outputs.clear();


      SpvReflectShaderModule module = {};
      SpvReflectResult result = spvReflectCreateShaderModule(
          spirv_code.size() * sizeof(uint32_t),
          spirv_code.data(),
          &module
      );
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
          return fail("spv reflect failed");
      }

      // 2. 解析输入变量
      uint32_t input_count = 0;
      result = spvReflectEnumerateInputVariables(&module, &input_count, nullptr);
      if (result == SPV_REFLECT_RESULT_SUCCESS && input_count > 0) {
          std::vector<SpvReflectInterfaceVariable*> input_vars(input_count);
          result = spvReflectEnumerateInputVariables(&module, &input_count, input_vars.data());
          if (result == SPV_REFLECT_RESULT_SUCCESS) {
              out_interface.inputs.reserve(input_count);
              for (uint32_t i = 0; i < input_count; ++i) {
                  out_interface.inputs.push_back(reflect_var_to_info(input_vars[i]));
              }
          }
      }

      // 3. 解析输出变量
      uint32_t output_count = 0;
      result = spvReflectEnumerateOutputVariables(&module, &output_count, nullptr);
      if (result == SPV_REFLECT_RESULT_SUCCESS && output_count > 0) {
          std::vector<SpvReflectInterfaceVariable*> output_vars(output_count);
          result = spvReflectEnumerateOutputVariables(&module, &output_count, output_vars.data());
          if (result == SPV_REFLECT_RESULT_SUCCESS) {
              out_interface.outputs.reserve(output_count);
              for (uint32_t i = 0; i < output_count; ++i) {
                  out_interface.outputs.push_back(reflect_var_to_info(output_vars[i]));
              }
          }
      }

      // 4. 排序
      sort_interface_variables(out_interface.inputs);
      sort_interface_variables(out_interface.outputs);

      // 5. 清理
      spvReflectDestroyShaderModule(&module);
      return out_interface;
  }


bool vulkan::pipeline::validate_interface_match(
    const shader_stage_interface& producer,
    const shader_stage_interface& consumer
  ){
  // 获取非内置变量的输出和输入
  std::vector<const interface_variable_info*> producer_outputs;
  std::vector<const interface_variable_info*> consumer_inputs;

  for (const auto& out : producer.outputs) {
    if (!out.is_builtin) producer_outputs.push_back(&out);
  }
  for (const auto& in : consumer.inputs) {
    if (!in.is_builtin) consumer_inputs.push_back(&in);
  }

  // 数量必须一致
  if (producer_outputs.size() != consumer_inputs.size()) {
    // 错误：输出/输入变量数量不匹配
    return false;
  }

  // 逐一比对
  for (size_t i = 0; i < producer_outputs.size(); ++i) {
    const vulkan::pipeline::interface_variable_info* p_out = producer_outputs[i];
    const vulkan::pipeline::interface_variable_info* p_in = consumer_inputs[i];

    if (p_out->location != p_in->location) {
      // 错误：location 不匹配
      return false;
    }
    if (p_out->component != p_in->component) {
      // 错误：component 不匹配
      return false;
    }
    if (p_out->format != p_in->format) {
      // 错误：format 不匹配
      return false;
    }
  }

  return true;
}

std::expected<std::vector<vulkan::pipeline::descriptor_set_layout_data>, std::string_view>
vulkan::pipeline::parse_descriptor_set_layouts(
    const std::span<unsigned char> spirv_code,
    const VkShaderStageFlagBits shader_stage
){

    using fail = std::unexpected<std::string_view>;

    std::vector<descriptor_set_layout_data> set_layouts;

      // 1. 创建并加载 Shader Module
      SpvReflectShaderModule module = {};
      SpvReflectResult result = spvReflectCreateShaderModule(
          spirv_code.size() * sizeof(uint32_t),
          spirv_code.data(),
          &module
      );
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
          return fail("spv reflect failed"); // 加载失败，返回空
      }

      // 2. 查询描述符集的数量
      uint32_t set_count = 0;
      result = spvReflectEnumerateDescriptorSets(&module, &set_count, nullptr);
      if (result != SPV_REFLECT_RESULT_SUCCESS || set_count == 0) {
          spvReflectDestroyShaderModule(&module);
          return set_layouts;
      }

      // 3. 获取所有描述符集的指针
      std::vector<SpvReflectDescriptorSet*> sets(set_count);
      result = spvReflectEnumerateDescriptorSets(&module, &set_count, sets.data());
      if (result != SPV_REFLECT_RESULT_SUCCESS) {
          spvReflectDestroyShaderModule(&module);
          return set_layouts;
      }

      // 4. 遍历并转换数据
      set_layouts.resize(sets.size());
      for (size_t i = 0; i < sets.size(); ++i) {
          const SpvReflectDescriptorSet& refl_set = *(sets[i]);
          vulkan::pipeline::descriptor_set_layout_data& layout_data = set_layouts[i];

          layout_data.set_number = refl_set.set;
          layout_data.bindings.resize(refl_set.binding_count);

          for (uint32_t j = 0; j < refl_set.binding_count; ++j) {
              const SpvReflectDescriptorBinding& refl_binding = *(refl_set.bindings[j]);
              VkDescriptorSetLayoutBinding& vk_binding = layout_data.bindings[j];

              vk_binding.binding = refl_binding.binding;
              vk_binding.descriptorType = static_cast<VkDescriptorType>(refl_binding.descriptor_type);

              // 计算描述符数量，处理数组情况
              vk_binding.descriptorCount = 1;
              for (uint32_t dim = 0; dim < refl_binding.array.dims_count; ++dim) {
                  vk_binding.descriptorCount *= refl_binding.array.dims[dim];
              }

              // 设置着色器阶段
              vk_binding.stageFlags = shader_stage;
          }

          // 填充最终的 VkDescriptorSetLayoutCreateInfo
          layout_data.create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
          layout_data.create_info.bindingCount = static_cast<uint32_t>(layout_data.bindings.size());
          layout_data.create_info.pBindings = layout_data.bindings.data();
      }

      // 5. 清理并返回
      spvReflectDestroyShaderModule(&module);
      return set_layouts;
  }

std::expected<vulkan::pipeline::push_constant_layout, std::string_view> vulkan::pipeline::parse_push_constant_layout(
    const std::span<unsigned char> spirv_code){

    using fail = std::unexpected<std::string_view>;
    push_constant_layout out_layout;
    out_layout.constants.clear();
    out_layout.total_size = 0;

    SpvReflectShaderModule module = {};
    SpvReflectResult result = spvReflectCreateShaderModule(
        spirv_code.size() * sizeof(uint32_t),
        spirv_code.data(),
        &module
    );
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        return fail("spv reflect failed");
    }

    uint32_t pc_count = 0;
    result = spvReflectEnumeratePushConstantBlocks(&module, &pc_count, nullptr);
    if (result != SPV_REFLECT_RESULT_SUCCESS || pc_count == 0) {
        spvReflectDestroyShaderModule(&module);
        return {}; // 没有 Push Constant 也是合法状态
    }

    std::vector<SpvReflectBlockVariable*> pc_blocks(pc_count);
    result = spvReflectEnumeratePushConstantBlocks(&module, &pc_count, pc_blocks.data());
    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        spvReflectDestroyShaderModule(&module);
        return fail("enumerate push constants failed");
    }

    out_layout.constants.reserve(pc_count);
    uint32_t max_offset = 0;

    for (uint32_t i = 0; i < pc_count; ++i) {
        const SpvReflectBlockVariable* p_block = pc_blocks[i];

        for (uint32_t j = 0; j < p_block->member_count; ++j) {
            const SpvReflectBlockVariable& member = p_block->members[j];

            push_constant_info info = {};
            info.offset = member.offset;
            info.size = member.size;
            info.stage_flags = static_cast<VkShaderStageFlags>(module.shader_stage);
            info.name = member.name ? member.name : "unnamed";

            out_layout.constants.push_back(info);

            const uint32_t end = member.offset + member.size;
            if (end > max_offset) {
                max_offset = end;
            }
        }
    }

    std::ranges::sort(out_layout.constants,
                      [](const push_constant_info& a, const push_constant_info& b) {
                          return a.offset < b.offset;
                      });

    out_layout.total_size = max_offset;

    // 6. 清理
    spvReflectDestroyShaderModule(&module);
    return out_layout;
}