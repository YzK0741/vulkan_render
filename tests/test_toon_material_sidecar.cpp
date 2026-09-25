// Headless unit tests: toon_material_sidecar (pure CPU) ==========================
// The `.toon.tsv` a model may carry beside it: the tab-separated format, the two rules the module exists to
// keep (a feature is switched on by an explicit `_Use` flag; a MISSING sidecar is not an error), and the
// failure cases - because a reader whose failure cases are not asserted is a reader whose failures are
// discovered by a character losing its ramp and looking slightly wrong.
//
// THE FAILURE CASES ARE THE TEST, as in test_render_resources: most of the checks below assert that something
// is REJECTED, and each one is a disagreement between this reader and the asset pipeline that would otherwise
// pass silently.
//
// THE LAST CHECK IS NOT ABOUT THE SIDECAR READER AT ALL, and it is here deliberately: the host bakes the ramp
// that fills a sidecar's `_DiffRampMap` LANE, and the stage that reads that lane inverts the bake using a
// constant of its own. The two numbers are written in two languages in two files, so their agreement is a
// contract nothing else can check - see `test_the_baked_ramp_and_the_shader_agree_on_its_width`.
#include "vk_test.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

import gltf_loader;
import toon_material_sidecar;

namespace {

    /// a well-formed sidecar, inline so the parser's own cases need no file on disk
    constexpr std::string_view good_text =
        "material\tkind\tname\tvalue\n"
        "M_body\tslot\t_DiffRampMap\tT_body_RD\n"
        "M_body\tfloat\t_UseDiffRampMap\t1.0\n"
        "M_body\tfloat\t_UseSpecRampMap\t0.0\n"
        "M_body\tfloat\t_OutlineWidth\t0.6\n"
        "M_body\tcolor\t_OutlineColor\t0.1,0.1,0.1,1.0\n"
        "M_hair\tslot\t_LineMap\tT_hair_line\n"
        "M_hair\tfloat\t_UseLineMap\t1.0\n";

    void test_well_formed() {
        auto const parsed = toon::parse_sidecar(good_text);
        CHECK(parsed.has_value());
        if (!parsed.has_value()) {
            return;
        }
        toon::sidecar const& sidecar = *parsed;
        CHECK(sidecar.materials.size() == 2);
        if (sidecar.materials.size() != 2) {
            return;
        }
        // THE FILE'S ORDER IS KEPT, and the names are the asset pipeline's
        CHECK(sidecar.materials[0].name == "M_body");
        CHECK(sidecar.materials[1].name == "M_hair");

        toon::material_sidecar const* const body = sidecar.find("M_body");
        CHECK(body != nullptr);
        if (body == nullptr) {
            return;
        }
        // a slot's value is the TEXTURE NAME as written, not a path and not an index
        CHECK(body->slot("_DiffRampMap") == "T_body_RD");
        // an absent slot is empty rather than a guess
        CHECK(body->slot("_SpecRampMap").empty());
        // a scalar reads back, and an absent one falls back
        CHECK(body->scalar("_OutlineWidth", -1.0f) == 0.6f);
        CHECK(body->scalar("_ShadowLutTex", -1.0f) == -1.0f);
        // a `color` row is KEPT rather than dropped, so an unknown kind reaches a consumer as data
        CHECK(body->others.size() == 1);
        CHECK(body->others.contains("_OutlineColor"));

        // THE FIRST RULE: the explicit flag decides, not the slot's presence.
        CHECK(body->enabled("_DiffRampMap"));  // _UseDiffRampMap = 1.0
        CHECK(!body->enabled("_SpecRampMap")); // _UseSpecRampMap = 0.0
        // ... and a flag that is ABSENT is OFF, which is the safe answer: an artist's switch is off unless
        // it was switched on, so a consumer cannot turn on a feature by finding a slot.
        CHECK(!body->enabled("_ShadowLutTex"));

        toon::material_sidecar const* const hair = sidecar.find("M_hair");
        CHECK(hair != nullptr);
        CHECK(hair != nullptr && hair->enabled("_LineMap"));
        // the name prefix resolves the same way for a slot that has no leading underscore
        CHECK(body->enabled("OutlineWidth") == false); // there is no _UseOutlineWidth, so: off
        CHECK(sidecar.find("M_absent") == nullptr);
    }

    void test_crlf_and_blank_lines() {
        // CRLF because the file is written on Windows, and blank lines because a hand-edited one has them
        constexpr std::string_view text =
            "material\tkind\tname\tvalue\r\n"
            "\r\n"
            "M_a\tslot\t_BaseMap\tT_a\r\n";
        auto const parsed = toon::parse_sidecar(text);
        CHECK(parsed.has_value());
        if (!parsed.has_value()) {
            return;
        }
        CHECK(parsed->materials.size() == 1);
        // THE \r IS STRIPPED FROM THE LAST FIELD TOO, which a naive split leaves attached and which would
        // make every texture name end in a carriage return
        CHECK(parsed->materials.size() == 1 && parsed->materials[0].slot("_BaseMap") == "T_a");
        CHECK(parsed->skipped_lines == 2); // the header and the blank line
    }

    void test_header_is_recognised_by_its_column_not_its_position() {
        // a file with NO header is still readable, which is what "recognised by its first column" buys
        constexpr std::string_view text = "M_a\tslot\t_BaseMap\tT_a\n";
        auto const parsed = toon::parse_sidecar(text);
        CHECK(parsed.has_value());
        CHECK(parsed.has_value() && parsed->materials.size() == 1 && parsed->materials[0].name == "M_a");
    }

    void test_a_malformed_row_is_an_error_with_its_line_number() {
        constexpr std::string_view text =
            "material\tkind\tname\tvalue\n"
            "M_a\tslot\t_BaseMap\tT_a\n"
            "M_a\tslot\t_TooFewColumns\n";
        auto const parsed = toon::parse_sidecar(text);
        CHECK(!parsed.has_value()); // THE POINT OF THE TEST: it is rejected, not skipped
        if (parsed.has_value()) {
            return;
        }
        // the message names the LINE, which is what makes a broken asset pipeline one line to read
        CHECK(parsed.error().find("line 3") != std::string::npos);
    }

    void test_a_non_numeric_float_is_an_error() {
        constexpr std::string_view text =
            "material\tkind\tname\tvalue\n"
            "M_a\tfloat\t_OutlineWidth\twide\n";
        auto const parsed = toon::parse_sidecar(text);
        CHECK(!parsed.has_value());
        if (parsed.has_value()) {
            return;
        }
        CHECK(parsed.error().find("_OutlineWidth") != std::string::npos);
        // ... and a number with trailing junk is not a number either
        auto const trailing = toon::parse_sidecar("M_a\tfloat\t_X\t1.0x\n");
        CHECK(!trailing.has_value());
    }

    void test_the_path_convention_appends_rather_than_replaces() {
        // `x.glb` -> `x.glb.toon.tsv`, NOT `x.toon.tsv`: the model's own extension stays
        std::filesystem::path const path = toon::sidecar_path_for("models/hero.glb");
        CHECK(path.filename() == "hero.glb.toon.tsv");
        CHECK(path.parent_path() == "models");
    }

    void test_a_missing_file_is_an_empty_sidecar_and_not_an_error() {
        // THE SECOND RULE: every non-character model has no sidecar, so this is the normal path rather than a
        // failure to handle - and the check is that it does NOT report one.
        auto const loaded = toon::load_sidecar(VR_TEST_SOURCE_DIR "/tests/fixtures/toon/absent.gltf");
        CHECK(loaded.has_value());
        CHECK(loaded.has_value() && loaded->empty());
        // an empty sidecar answers "no" to every lookup rather than asserting
        CHECK(loaded.has_value() && loaded->find("M_anything") == nullptr);
    }

    void test_a_file_on_disk_is_read_through_the_convention() {
        auto const loaded = toon::load_sidecar(VR_TEST_SOURCE_DIR "/tests/fixtures/toon/minimal.gltf");
        CHECK(loaded.has_value());
        if (!loaded.has_value()) {
            return;
        }
        CHECK(loaded->materials.size() == 2);
        toon::material_sidecar const* const body = loaded->find("M_actor_test_body_01");
        CHECK(body != nullptr);
        if (body == nullptr) {
            return;
        }
        CHECK(body->slot("_BaseMap") == "T_actor_test_body_01_D");
        CHECK(body->slot("_DiffRampMap") == "T_actor_common_body_01_RD");
        // THE CASE A CONSUMER THAT INFERRED WOULD GET WRONG: `_BumpMap` IS DECLARED and `_UseBumpMap` is 0.0,
        // so the slot is present while the feature is OFF. "Declared" and "enabled" are two different
        // questions, which is the first rule of the module's header and the reason the flag exists at all.
        CHECK(!body->slot("_BumpMap").empty());
        CHECK(!body->enabled("_BumpMap"));
        CHECK(body->enabled("_DiffRampMap"));
        // ... and `_SpecRampMap` is neither declared nor enabled, so the two really do vary independently
        CHECK(body->slot("_SpecRampMap").empty());
        CHECK(!body->enabled("_SpecRampMap"));
    }

    void test_a_malformed_file_on_disk_reports_its_path() {
        auto const loaded = toon::load_sidecar(VR_TEST_SOURCE_DIR "/tests/fixtures/toon/broken.gltf");
        CHECK(!loaded.has_value());
        if (loaded.has_value()) {
            return;
        }
        // the path is in the message, because a sidecar that cannot be parsed is a file to go and look at
        CHECK(loaded.error().find("broken.gltf.toon.tsv") != std::string::npos);
    }

    void test_the_sidecar_and_the_model_join_by_texture_name() {
        // THE JOIN THE TWO MODULES EXIST FOR, and it is the whole reason the loader carries image names: the
        // sidecar refers to a toon map by bare asset name, and the model's IMAGE NAMES are what turn that into
        // something loadable. Nothing else connects the two files.
        auto const model = gltf::load_model(VR_TEST_SOURCE_DIR "/tests/fixtures/toon/named_texture.gltf");
        CHECK(model.has_value());
        if (!model.has_value()) {
            return;
        }
        auto const sidecar = toon::load_sidecar(VR_TEST_SOURCE_DIR "/tests/fixtures/toon/named_texture.gltf");
        CHECK(sidecar.has_value());
        if (!sidecar.has_value()) {
            return;
        }
        toon::material_sidecar const* const body = sidecar->find("M_actor_test_body_01");
        CHECK(body != nullptr);
        if (body == nullptr) {
            return;
        }
        CHECK(model->textures.size() == 2);
        std::optional<uint16_t> const base = model->texture_index_by_name(body->slot("_BaseMap"));
        std::optional<uint16_t> const ramp = model->texture_index_by_name(body->slot("_DiffRampMap"));
        CHECK(base.has_value());
        CHECK(ramp.has_value());
        // two DIFFERENT images, so the join really resolved names rather than returning a constant
        CHECK(base.has_value() && ramp.has_value() && *base != *ramp);

        // A NAME THE MODEL DOES NOT HAVE MISSES, AND THAT IS A STATE RATHER THAN AN ERROR: the sidecar turns a
        // feature ON for a texture the artist did not export, which happens whenever an optional map is absent.
        // The consumer's safe answer is to leave the feature off - which is the same rule as the `_Use` flag,
        // arriving from the other side.
        CHECK(body->enabled("_SpecRampMap"));
        CHECK(!model->texture_index_by_name(body->slot("_SpecRampMap")).has_value());

        // the round trip: a texture that HAS a name is found by it
        CHECK(model->texture_index_by_name(model->textures[0].name).has_value());
        CHECK(model->texture_index_by_name(model->textures[0].name) == std::optional<uint16_t>{0});
        // AN EMPTY NAME MATCHES NOTHING, which is what keeps a model whose images are all unnamed (every other
        // model in this repository) from resolving every query to its first texture
        CHECK(!model->texture_index_by_name("").has_value());
    }

    /// the value of the first `<name> = <number>` in @p text, or nothing - a five-line parser, because a regex
    /// or a build dependency would be more machinery than one shared constant is worth
    std::optional<float> float_constant_of(std::string_view const text, std::string_view const name) {
        std::string const needle = std::string(name) + " =";
        std::size_t const at = text.find(needle);
        if (at == std::string_view::npos) {
            return std::nullopt;
        }
        char const* const begin = text.data() + at + needle.size();
        char* end = nullptr;
        float const value = std::strtof(begin, &end);
        if (end == begin) {
            return std::nullopt;
        }
        return value;
    }

    void test_the_baked_ramp_and_the_shader_agree_on_its_width() {
        // THE ONE NUMBER THE BAKE AND THE READ SHARE, and why it needs a test rather than a comment on each
        // side: the host bakes a neutral step at x = 0.5 with half width `w`, and the shader inverts that bake
        // by reading at `0.5 + (gated - center) * (w' / softness)`. When `w != w'` the ramp STILL LOOKS LIKE A
        // RAMP - a step is a step - so nothing about the frame announces that the contract broke; what happens
        // is that the texture branch and the procedural branch stop agreeing and each family's terminator sits
        // at the wrong place. That is the silent-drift shape this project writes tests for.
        //
        // IT LIVES HERE rather than in a test of its own because it is a contract between the host's sidecar
        // LANE bake and the stage that reads that lane, which is what this test is about - and because a test
        // target of its own would have to be mirrored into VR_TEST_TARGETS, the workflow and the docs.
        std::ifstream host_file{VR_TEST_SOURCE_DIR "/main.cpp"};
        std::ifstream shader_file{VR_TEST_SOURCE_DIR "/shaders/character_forward.slang"};
        CHECK(host_file.good());
        CHECK(shader_file.good());
        if (!host_file.good() || !shader_file.good()) {
            return;
        }
        std::string const host{std::istreambuf_iterator<char>{host_file}, std::istreambuf_iterator<char>{}};
        std::string const shader{std::istreambuf_iterator<char>{shader_file}, std::istreambuf_iterator<char>{}};

        std::optional<float> const baked = float_constant_of(host, "baked_ramp_half_width");
        std::optional<float> const read_at = float_constant_of(shader, "character_ramp_half_width");
        CHECK(baked.has_value());
        CHECK(read_at.has_value());
        CHECK(baked == read_at);

        // AND THE BAKE PUTS THE STEP AT THE CENTRE THE SHADER'S REMAP ASSUMES. `0.5` is written into both the
        // bake's `smoothstep` call and the shader's remap, so asserting the two SPELLINGS is what keeps a bake
        // whose step sat anywhere else from shifting every family's terminator by the difference while both
        // constants above still matched.
        CHECK(host.find("smoothstep(0.5f - baked_ramp_half_width, 0.5f + baked_ramp_half_width") != std::string::npos);
        CHECK(shader.find("saturate(0.5 + (gated - params.center)") != std::string::npos);

        // BOTH RAMP LANES ARE ON THE SAME CONTRACT, and this assertion is what keeps the second one from being
        // wired with a constant of its own: the specular lane's remap has to divide by the SAME
        // `character_ramp_half_width`, because the host bakes ONE asset for the two lanes and cannot invert it
        // at two different widths.
        CHECK(shader.find("saturate(0.5 + (no_h - params.spec_center) * (character_ramp_half_width") != std::string::npos);

        // THE SHADOW LUT'S TILE COUNT IS THE SAME KIND OF CONTRACT, and it fails the same silent way: the bake
        // lays a `32^3` cube into a `1024x32` strip of `32x32` tiles and the shader's `toon_shadow_lut` inverts
        // that layout, so a bake for a DIFFERENT tile count reads back as a different COLOUR - the lane still
        // answers, it just answers with the wrong cube - and nothing about the frame says which side moved.
        std::optional<float> const lut_tiles = float_constant_of(host, "baked_lut_tiles");
        std::optional<float> const lut_tiles_read = float_constant_of(shader, "character_shadow_lut_tiles");
        CHECK(lut_tiles.has_value());
        CHECK(lut_tiles_read.has_value());
        CHECK(lut_tiles == lut_tiles_read);
    }

} // namespace

int main() {
    test_well_formed();
    test_crlf_and_blank_lines();
    test_header_is_recognised_by_its_column_not_its_position();
    test_a_malformed_row_is_an_error_with_its_line_number();
    test_a_non_numeric_float_is_an_error();
    test_the_path_convention_appends_rather_than_replaces();
    test_a_missing_file_is_an_empty_sidecar_and_not_an_error();
    test_a_file_on_disk_is_read_through_the_convention();
    test_a_malformed_file_on_disk_reports_its_path();
    test_the_sidecar_and_the_model_join_by_texture_name();
    test_the_baked_ramp_and_the_shader_agree_on_its_width();
    return vk_test::finish("test_toon_material_sidecar");
}
