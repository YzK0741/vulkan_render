import std;
import vulkan.runtime;

namespace {
    // 以二进制方式读取整个文件；失败返回 false
    bool read_binary_file(const std::filesystem::path& path, std::vector<unsigned char>& out) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return false;
        }
        out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        return !file.bad();
    }

    // 从当前工作目录向上逐级查找 shaders/ 目录，
    // 兼容在项目根目录或 cmake-build-* 目录下运行的情况
    std::optional<std::filesystem::path> locate_shaders_dir() {
        std::filesystem::path current = std::filesystem::current_path();
        for (int depth = 0; depth < 4; ++depth) {
            const std::filesystem::path candidate = current / "shaders";
            if (std::filesystem::is_directory(candidate)) {
                return candidate;
            }
            const std::filesystem::path parent = current.parent_path();
            if (parent == current) {
                break;
            }
            current = parent;
        }
        return std::nullopt;
    }

    // 读取单个着色器 SPIR-V 文件并打印信息；失败返回 false
    bool load_shader(const std::filesystem::path& dir, std::string_view file_name, std::vector<unsigned char>& out) {
        const std::filesystem::path path = dir / file_name;
        if (!read_binary_file(path, out)) {
            std::println("ERROR: cannot open shader file '{}'", path.string());
            return false;
        }
        std::println("loaded shader: {} ({} bytes)", path.string(), out.size());
        return true;
    }

    // 加载一对 vertex/fragment SPIR-V，并通过 runtime 创建管线
    bool load_and_create_pipeline(vulkan::runtime& runtime,
                                  const std::filesystem::path& shaders_dir,
                                  std::string_view pipeline_name,
                                  std::string_view vertex_file,
                                  std::string_view fragment_file) {
        std::vector<unsigned char> vertex_code;
        std::vector<unsigned char> fragment_code;
        if (!load_shader(shaders_dir, vertex_file, vertex_code) ||
            !load_shader(shaders_dir, fragment_file, fragment_code)) {
            return false;
        }

        const std::expected<void, std::string> result = runtime.make_pipeline(pipeline_name, vertex_code, fragment_code);
        if (!result) {
            std::println("FAILED to create pipeline '{}': {}", pipeline_name, result.error());
            return false;
        }
        std::println("SUCCESS: pipeline '{}' created and cached in the runtime", pipeline_name);
        return true;
    }
} // namespace

int main() {
    // 1. 定位 shaders 目录（保存 GLSL 源码与编译好的 SPIR-V）
    const std::optional<std::filesystem::path> shaders_dir = locate_shaders_dir();
    if (!shaders_dir) {
        std::println("ERROR: cannot find shaders/ directory. run the program from the project root or a cmake-build-* directory.");
        return 1;
    }

    // 2. 构造 vulkan::runtime：默认构造会完成 window/instance/device/swapchain 等全部初始化
    vulkan::runtime runtime;

    // 3. 依次加载并创建两条管线：简单三角形 + 标准 PBR
    bool all_ok = true;
    if (!load_and_create_pipeline(runtime, *shaders_dir, "triangle", "triangle.vert.spv", "triangle.frag.spv")) {
        all_ok = false;
    }
    if (!load_and_create_pipeline(runtime, *shaders_dir, "pbr", "pbr.vert.spv", "pbr.frag.spv")) {
        all_ok = false;
    }
    if (!all_ok) {
        std::println("hint: compile the shaders first: powershell -ExecutionPolicy Bypass -File shaders/compile_shaders.ps1");
        return 1;
    }

    std::println("ALL PIPELINES LOADED SUCCESSFULLY");
    return 0;
}
