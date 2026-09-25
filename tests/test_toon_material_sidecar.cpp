// Headless unit tests: toon_material_sidecar (pure CPU) ==========================
// The `.toon.tsv` a model may carry beside it: the tab-separated format, the two rules the module exists to
// keep (a feature is switched on by an explicit `_Use` flag; a MISSING sidecar is not an error), and the
// failure cases - because a reader whose failure cases are not asserted is a reader whose failures are
// discovered by a character losing its ramp and looking slightly wrong.
//
// THE FAILURE CASES ARE THE TEST, as in test_render_resources: most of the checks below assert that something
// is REJECTED, and each one is a disagreement between this reader and the asset pipeline that would otherwise
// pass silently.
#include "vk_test.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

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
    return vk_test::finish("test_toon_material_sidecar");
}
