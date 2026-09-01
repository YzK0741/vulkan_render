export module vulkan.math;
export import std;

/**
 * @file math.cppm
 * @defgroup vulkan_math Vulkan Math (CPU-side)
 * @brief CPU-side math for the renderer: IBL precomputation (prefiltered environment mip chain,
 *        irradiance cubemap, BRDF LUT) and half-float conversion
 * @note
 *      - pure CPU math, no Vulkan or GPU resources involved
 *      - moved out of vulkan.model so that module stays focused on GPU resource management
 */
namespace vulkan {

    /**
     * @ingroup vulkan_math
     * @brief generate a procedural HDR environment cubemap (RGBA32F, 6 faces packed)
     */
    export std::vector<float> generate_environment_cubemap(int size);

    /**
     * @ingroup vulkan_math
     * @brief GGX importance-sampled prefilter of the environment into a mip chain
     * @param env the base environment cubemap from generate_environment_cubemap() (read-only view)
     * @param env_size base cubemap size
     * @param mip_count number of mip levels
     * @return mip-major RGBA32F data (mip0 all faces, then mip1, ...), ready for a cubemap upload
     */
    export std::vector<float> prefilter_environment(std::span<const float> env, int env_size, int mip_count);

    /**
     * @ingroup vulkan_math
     * @brief cosine-weighted hemisphere convolution for the diffuse irradiance cubemap
     */
    export std::vector<float> generate_irradiance_map(std::span<const float> env, int env_size, int irr_size);

    /**
     * @ingroup vulkan_math
     * @brief BRDF integration LUT filled with the Frostbite analytic approximation (RG32F: scale, bias)
     */
    export std::vector<float> generate_brdf_lut(int size);

    /**
     * @ingroup vulkan_math
     * @brief convert 4-channel float data to a packed RGBA16F byte stream
     */
    export std::vector<unsigned char> to_half_rgba(std::span<const float> data);

    /**
     * @ingroup vulkan_math
     * @brief convert 2-channel float data to a packed RG16F byte stream
     */
    export std::vector<unsigned char> to_half_rg(std::span<const float> data);
} // namespace vulkan
