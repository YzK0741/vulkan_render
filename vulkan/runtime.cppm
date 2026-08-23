//
// Created by 小叶 on 2026/8/2.
//

module;

#include <vulkan/vulkan.h>

export module vulkan.runtime;

export import vulkan.core;

/**
 * @file runtime.cppm
 * @defgroup vulkan_runtime Vulkan Runtime Facade
 * @brief runtime facade: a thin wrapper exposing all functionality of vulkan::core
 * @note
 *      - access the inner core via operator->, same usage as a raw core
 *      - the inner core's lifetime is tied to the runtime
 */
namespace vulkan {
    /**
     * @ingroup vulkan_runtime
     * @brief vulkan runtime facade class
     * @note
     *      - use operator-> to access the inner core (e.g. runtime->vma.create_buffer(...))
     *      - default construction performs the whole core initialization (window/instance/device/swap chain etc.)
     */
    export class runtime {
        core vulkan_core = {};

    public:
        core* operator->() {
            return &this->vulkan_core;
        }
    };
}