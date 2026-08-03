//
// Created by 小叶 on 2026/8/2.
//

module;

#include <vulkan/vulkan.h>

export module vulkan.runtime;

export import vulkan.core;

namespace vulkan {
    export class runtime {
        core vulkan_core = {};

    public:
        core* operator->() {
            return &this->vulkan_core;
        }
    };
}