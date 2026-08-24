module vulkan.runtime;

namespace vulkan
{
    std::expected<void, std::string> runtime::make_pipeline(std::string_view pipeline_name, std::span<const unsigned char> vertex_shader_code, std::span<const unsigned char> fragment_shader_code)
    {
        using fail = std::unexpected<std::string>;
        auto make_result = this->vulkan_core.make_pipeline(vertex_shader_code, fragment_shader_code);
        if (!make_result)
        {
            return fail(make_result.error());
        }
        this->pipelines.emplace(pipeline_name, std::move(make_result).value());
        return {};
    }
}
