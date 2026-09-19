#!/usr/bin/env python3
"""Minimal PMX (MMD) -> GLB converter, for evaluating NPR shading in this renderer.

SHADING REFERENCE AND CREDIT. The NPR look this asset is being prepared for follows the Zenless Zone Zero
shader by **新杨XIYAG** (bilibili), as redistributed with the rig by Crabnuts and optimised by Luci in
https://github.com/fnoji/Blender-ZZZ-XIYAG-Shader (XIYAG_ZZZ_Shader.blend). That shader's own instructions
are what this converter serves: THREE UV maps (UV0/UV1/UV2), which is why the additional PMX UV sets are
exported as TEXCOORD_1 / TEXCOORD_2 rather than discarded. Per that project's rules, any work that uses the
shader or takes it as a primary reference credits **新杨XIYAG**. The same credit appears at the top of the
NPR shaders in shaders/.

WHY THIS EXISTS. The renderer's glTF loader is the only asset path it has, and the model we want to look
at is a PMX. Blender is not installed on this machine; Python 3.12 is. So this reads the parts of PMX 2.0
that a *displayed* model needs and writes a GLB.

WHAT IT DELIBERATELY DOES NOT READ, and the reason it can stop early: a PMX is stored in this order -
header, vertices, faces, textures, materials, bones, morphs, display frames, rigid bodies, joints - and
everything we need is in the first five sections. Bones, morphs and physics are never parsed, so no
skinning data is emitted either: the model comes out in its BIND POSE, which is what a shading comparison
needs. (Implementing skinning would mean mapping BDEF1/BDEF2/BDEF4 to glTF's four joints/weights and
approximating SDEF, which cannot be represented exactly - a separate job, and not one this evaluation
depends on.)

WHAT IT CARRIES THAT MATTERS FOR NPR, because these are the model's authored shading inputs:
  * the diffuse texture per material, plus its diffuse colour and sphere/toon modes as extras;
  * MMD's per-vertex EDGE SCALE, which is the outline thickness, written as `_EDGESCALE` - a SCALAR float
    with a leading underscore, i.e. the glTF spec's name for an application-specific attribute. NOT COLOR_0,
    which is where this started: the spec defines COLOR_0 as a MULTIPLIER of the base colour, so a
    conforming viewer tints the whole model by it - measured in the Khronos glTF Sample Viewer, which drew
    this asset solid RED, because the edge scale was written as (scale, 0, 0, 1) and its green and blue
    lanes are zero. The reference asset for the same shading (Endmin) carries `_EDGESCALE` the same way,
    which is the other reason to match it.
  * the SPEHRE/TOON texture indices as glTF extras, so a shader can find `sp.png` / `toon_defo.bmp`.

TEXTURES: glTF allows PNG and JPEG only, and MMD assets are full of BMP. BMPs are re-encoded to PNG here
with zlib (a non-interlaced truecolor PNG is about twenty lines and needs no dependency).

USAGE:  python pmx_to_glb.py <model.pmx> <out.glb>
"""

from __future__ import annotations

import json
import math
import struct
import sys
import zlib
from pathlib import Path

# --------------------------------------------------------------------------------------------- reading


class Reader:
    def __init__(self, data: bytes, utf16: bool) -> None:
        self.data = data
        self.pos = 0
        self.utf16 = utf16

    def bytes(self, n: int) -> bytes:
        out = self.data[self.pos : self.pos + n]
        if len(out) != n:
            raise EOFError(f"wanted {n} bytes at {self.pos}, got {len(out)}")
        self.pos += n
        return out

    def u8(self) -> int:
        return self.bytes(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.bytes(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.bytes(2))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.bytes(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.bytes(4))[0]

    def vec(self, n: int) -> tuple[float, ...]:
        return struct.unpack("<" + "f" * n, self.bytes(4 * n))

    def text(self) -> str:
        n = self.i32()
        raw = self.bytes(n)
        return raw.decode("utf-16-le" if self.utf16 else "utf-8", errors="replace")


def read_pmx(path: Path):
    data = path.read_bytes()
    if data[:4] != b"PMX ":
        raise ValueError("not a PMX file")
    version = struct.unpack("<f", data[4:8])[0]
    globals_count = data[8]
    globals_ = data[9 : 9 + globals_count]
    utf16 = globals_[0] == 0  # 0 = UTF-16LE, 1 = UTF-8
    uv_count = globals_[1]
    vertex_index_size = globals_[2]
    texture_index_size = globals_[3]
    material_index_size = globals_[4]
    bone_index_size = globals_[5]
    morph_index_size = globals_[6]
    rigid_body_index_size = globals_[7]

    r = Reader(data, utf16)
    r.pos = 9 + globals_count
    model_name = r.text()
    r.text()  # model name (english)
    r.text()  # comment
    r.text()  # comment (english)

    def index(size: int) -> int:
        if size == 1:
            return r.u8()
        if size == 2:
            return r.u16()
        return r.i32()

    # ---- vertices ----
    vertex_count = r.i32()
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    extra_uvs: list[list[tuple[float, float]]] = []  # PMX additional vec4 UVs -> glTF TEXCOORD_1/2
    edge_scales: list[float] = []
    for _ in range(vertex_count):
        position = r.vec(3)
        normal = r.vec(3)
        # PMX is LEFT-handed with its models facing -Z, and glTF is right-handed with the camera on +Z, so
        # the coordinates are MIRRORED in Z rather than rotated. A rotation would have preserved the
        # handedness and left the model's own left/right swapped; a mirror both turns it to face the camera
        # (measured: without this the model renders back-to-front, in a scene whose glTF models face it) and
        # converts the handedness. A mirror inverts triangle winding, which is why the faces are reversed.
        positions.append((position[0], position[1], -position[2]))
        # ... and the NORMAL is normalized, which the spec requires rather than merely prefers: glTF says a
        # normal attribute's vectors are unit length, and the Khronos validator reports
        # ACCESSOR_VECTOR3_NON_UNIT for every one that is not - 616 errors on this asset before this line
        # existed, because MMD's own normals are close to unit but not unit (633 of 32307 vertices sit
        # outside 1 +/- 0.0005, the smallest at 0.9914). A viewer that shades with them gets a slightly
        # wrong N.L and an IBL lookup in a slightly wrong direction.
        mirrored_normal = (normal[0], normal[1], -normal[2])
        normal_length = math.sqrt(mirrored_normal[0] ** 2 + mirrored_normal[1] ** 2 + mirrored_normal[2] ** 2)
        normals.append(
            (mirrored_normal[0] / normal_length, mirrored_normal[1] / normal_length, mirrored_normal[2] / normal_length)
            if normal_length > 1e-8
            else (0.0, 0.0, 1.0)  # a degenerate source normal has no direction to keep; any unit one will do
        )
        uv = r.vec(2)
        # NOT flipped, and that is a measurement rather than a preference. glTF's UV origin is the texture's
        # UPPER-left corner and so is MMD's, so the two agree and a flip mirrors every lookup. Measured on
        # this model's face material: with `1.0 - v` the forehead and the eyes sample the atlas's dark-fabric
        # and eye-closeup regions - a brown smear across the face, and black where the ornament is - while
        # without it the face is the face: irises, eyebrows, blush and the hair ornament where they were
        # authored. (This converter carried the flip from the start, with the comment that MMD's V is flipped
        # against glTF's; the render says otherwise.)
        uvs.append((uv[0], uv[1]))
        # The field is the count of ADDITIONAL vec4 UVs (the main UV is always present). They are KEPT: the
        # ZZZ-style shading this model is headed for needs UV0/UV1/UV2 - the XIYAG reference shader's own
        # instructions require three UV maps, see the credit note in the file header.
        extra: list[tuple[float, float]] = []
        for _ in range(uv_count):
            quad = r.vec(4)
            extra.append((quad[0], quad[1]))
        extra_uvs.append(extra)
        weight_type = r.u8()
        if weight_type == 0:  # BDEF1
            index(bone_index_size)
        elif weight_type == 1:  # BDEF2
            index(bone_index_size)
            index(bone_index_size)
            r.f32()
        elif weight_type == 2:  # BDEF4
            for _ in range(4):
                index(bone_index_size)
            for _ in range(4):
                r.f32()
        elif weight_type == 3:  # SDEF - approximated by ignoring the C/R0/R1 terms
            index(bone_index_size)
            index(bone_index_size)
            r.f32()
            r.vec(3)
            r.vec(3)
            r.vec(3)
        elif weight_type == 4:  # QDEF (PMX 2.1)
            for _ in range(4):
                index(bone_index_size)
            for _ in range(4):
                r.f32()
        edge_scales.append(r.f32())

    # ---- faces ----
    face_index_count = r.i32()
    face_indices = [index(vertex_index_size) for _ in range(face_index_count)]
    # Reversed within every triangle, because the Z mirror above inverts the winding and a reader is
    # entitled to cull by it. The model's materials are double-sided, so this is not what makes it visible -
    # it is what keeps the winding consistent with the coordinates.
    indices = [face_indices[tri + 2 - corner] for tri in range(0, len(face_indices), 3) for corner in range(3)]

    # ---- textures ----
    texture_count = r.i32()
    textures = [r.text() for _ in range(texture_count)]

    # ---- materials (read this far and no further) ----
    material_count = r.i32()
    materials = []
    for _ in range(material_count):
        name = r.text()
        r.text()  # name (english)
        diffuse = r.vec(4)
        specular = r.vec(3)
        specularity = r.f32()
        ambient = r.vec(3)
        draw_flags = r.u8()
        edge_color = r.vec(4)
        edge_size = r.f32()
        texture_index = index(texture_index_size)
        sphere_index = index(texture_index_size)
        sphere_mode = r.u8()  # 0 none, 1 multiply, 2 add, 3 sub-texture
        toon_flag = r.u8()  # 0 = texture reference, 1 = internal toon
        toon_index = index(texture_index_size) if toon_flag == 0 else r.u8()
        r.text()  # memo
        face_count = r.i32()
        materials.append(
            dict(
                name=name,
                diffuse=diffuse,
                specular=specular,
                specularity=specularity,
                ambient=ambient,
                draw_flags=draw_flags,
                edge_color=edge_color,
                edge_size=edge_size,
                texture_index=texture_index,
                sphere_index=sphere_index,
                sphere_mode=sphere_mode,
                toon_flag=toon_flag,
                toon_index=toon_index,
                face_count=face_count,
            )
        )

    return dict(
        version=version,
        name=model_name,
        positions=positions,
        normals=normals,
        uvs=uvs,
        extra_uvs=extra_uvs,
        edge_scales=edge_scales,
        indices=indices,
        textures=textures,
        materials=materials,
        uv_count=uv_count,
    )


# --------------------------------------------------------------------------------------------- writing


def png_from_bmp(raw: bytes) -> bytes | None:
    """Truecolor 24/32-bit BMP -> PNG. Returns None for anything else (the caller then skips it)."""
    if raw[:2] != b"BM":
        return None
    pixel_offset = struct.unpack("<I", raw[10:14])[0]
    header_size = struct.unpack("<I", raw[14:18])[0]
    if header_size < 40:
        return None
    width, height = struct.unpack("<ii", raw[18:26])
    bpp = struct.unpack("<H", raw[28:30])[0]
    compression = struct.unpack("<I", raw[30:34])[0]
    if compression != 0 or bpp not in (24, 32):
        return None
    flip = height > 0
    height = abs(height)
    stride = ((width * bpp // 8) + 3) & ~3
    channels = bpp // 8
    rows = []
    for y in range(height):
        src = pixel_offset + (height - 1 - y if flip else y) * stride
        row = bytearray()
        for x in range(width):
            px = raw[src + x * channels : src + x * channels + channels]
            b, g, r = px[0], px[1], px[2]
            row += bytes((r, g, b)) if channels == 3 else bytes((r, g, b, px[3]))
        rows.append(bytes(row))
    color_type = 2 if channels == 3 else 6
    out = bytearray(b"\x89PNG\r\n\x1a\n")

    def chunk(tag: bytes, payload: bytes) -> None:
        out.extend(struct.pack(">I", len(payload)))
        out.extend(tag)
        out.extend(payload)
        out.extend(struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    raw_scan = b"".join(b"\x00" + row for row in rows)
    chunk(b"IDAT", zlib.compress(raw_scan, 6))
    chunk(b"IEND", b"")
    return bytes(out)


def write_glb(out_path: Path, model: dict) -> dict:
    blob = bytearray()
    buffer_views = []
    accessors = []

    def add_view(payload: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(payload)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    positions = model["positions"]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    mins = [min(xs), min(ys), min(zs)]
    maxs = [max(xs), max(ys), max(zs)]

    def accessor(view: int, kind: str, count: int, comps: int, bbox: bool = False) -> int:
        acc = {
            "bufferView": view,
            "componentType": 5126 if kind == "f" else 5125,
            "count": count,
            "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[comps],
        }
        if bbox:
            acc["min"] = mins
            acc["max"] = maxs
        accessors.append(acc)
        return len(accessors) - 1

    pos_view = add_view(struct.pack("<" + "f" * (3 * len(positions)), *[c for p in positions for c in p]), 34962)
    pos_acc = accessor(pos_view, "f", len(positions), 3, bbox=True)
    nrm_view = add_view(struct.pack("<" + "f" * (3 * len(model["normals"])), *[c for n in model["normals"] for c in n]), 34962)
    nrm_acc = accessor(nrm_view, "f", len(model["normals"]), 3)
    uv_view = add_view(struct.pack("<" + "f" * (2 * len(model["uvs"])), *[c for t in model["uvs"] for c in t]), 34962)
    uv_acc = accessor(uv_view, "f", len(model["uvs"]), 2)
    # The per-vertex edge scale, as `_EDGESCALE`: ONE float per vertex, which is the form the reference
    # asset uses (see the header's note on why this is not COLOR_0).
    edge_view = add_view(struct.pack("<" + "f" * len(model["edge_scales"]), *model["edge_scales"]), 34962)
    edge_acc = accessor(edge_view, "f", len(model["edge_scales"]), 1)
    # the additional UV sets, as glTF TEXCOORD_1 / TEXCOORD_2 (VEC2: the first two components of each vec4)
    extra_acc = []
    for set_index in range(len(model["extra_uvs"][0]) if model["extra_uvs"] else 0):
        flat = [c for per_vertex in model["extra_uvs"] for c in per_vertex[set_index]]
        view = add_view(struct.pack("<" + "f" * len(flat), *flat), 34962)
        extra_acc.append(accessor(view, "f", len(model["extra_uvs"]), 2))
    index_view = add_view(struct.pack("<" + "I" * len(model["indices"]), *model["indices"]), 34963)
    index_acc = accessor(index_view, "u", len(model["indices"]), 1)

    # ---- textures: embedded in the same blob, BMP re-encoded to PNG ----
    images = []
    for rel in model["textures"]:
        path = (Path(model["_dir"]) / rel.replace("\\", "/")).resolve()
        if not path.exists():
            images.append(None)
            continue
        raw = path.read_bytes()
        mime = "image/png"
        if path.suffix.lower() == ".bmp":
            converted = png_from_bmp(raw)
            if converted is None:
                images.append(None)
                continue
            raw = converted
        elif path.suffix.lower() in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        view = add_view(raw)
        images.append((len(images), view, mime))
    real_images = [i for i in images if i is not None]
    image_index_of = {}
    gltf_images = []
    for slot, (_, view, mime) in enumerate(real_images):
        gltf_images.append({"bufferView": view, "mimeType": mime})
        image_index_of[slot] = len(gltf_images) - 1
    # remap: the PMX texture index -> the glTF image index
    pmx_to_gltf_image = {}
    slot = 0
    for pmx_index, image in enumerate(images):
        if image is None:
            continue
        pmx_to_gltf_image[pmx_index] = image_index_of[slot]
        slot += 1

    gltf_textures = [{"source": i} for i in range(len(gltf_images))]

    # The face block of an MMD model, by the name prefixes every PMX shares (see the extras below).
    MMD_FACE_MATERIAL_PREFIXES = (
        "顔",  # face skin
        "颜",  # ... the simplified form
        "口",  # mouth
        "齿",  # teeth
        "歯",
        "舌",  # tongue
        "睫",  # lashes
        "眉",  # brows
        "目",  # eyes (iris, highlight, shadow)
        "白目",  # ... and the eye whites, which start with 白 rather than 目
        "淚",  # tears
    )

    # The materials this engine draws from their ALBEDO rather than from the lighting stack: the
    # reference's `- Face` treatment, generalised to the head's own painted props. 千夏's ears live in
    # `头饰` - its geometry reaches y 18.842, the tallest thing in the model and above the hair - and an
    # ear lit like a surface reads as a lump of plastic where a drawn ear should read as a shape.
    # OPTION 2: the face is NOT painted any more - it is LIT again, with its own flattened shading normal
    # (see gather_surface) and its own cast-shadow retention, which is what the reference's `- Face`
    # group stands in for. This set is therefore only the head's PROPS.
    MMD_UNLIT_MATERIAL_PREFIXES = (
        "耳",  # ears
        "头饰",  # ... and the head wear they are modelled in on this asset
    )

    gltf_materials = []
    for m in model["materials"]:
        d = m["diffuse"]
        mat = {
            "name": m["name"],
            "pbrMetallicRoughness": {
                "baseColorFactor": [d[0], d[1], d[2], d[3]],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": (m["draw_flags"] & 0x01) != 0,
            "extras": {
                "mmd_edge_color": list(m["edge_color"]),
                "mmd_edge_size": m["edge_size"],
                "mmd_sphere_mode": m["sphere_mode"],
                "mmd_texture_name": model["textures"][m["texture_index"]] if 0 <= m["texture_index"] < len(model["textures"]) else None,
                "mmd_sphere_name": model["textures"][m["sphere_index"]] if 0 <= m["sphere_index"] < len(model["textures"]) else None,
                # ... AND ITS glTF TEXTURE INDEX, which is what the loader can actually bind: the name above
                # is for a human reading the file, and a loader would have to re-derive the mapping the
                # converter already has. Written only when the sphere texture was embedded (the BMP -> PNG
                # step can fail on a missing file), so "no key" and "no sphere" mean the same thing.
                "mmd_sphere_texture": pmx_to_gltf_image.get(m["sphere_index"]),
                # ... AND whether this is a FACE material. The reference shades the face with a separate
                # shader (its `- Face` group) whose shadow is far lighter than the body's, because a face
                # is nearly flat and its shading is painted into the texture; this is the flag that lets the
                # engine do the same. The match is by NAME because MMD names are the only place that fact
                # lives, and the prefixes below are the face block every PMX shares (skin, mouth, teeth,
                # lashes, brows, eyes, eye shadow) - checked against this asset's 22 materials, where they
                # select 0..11 and nothing else.
                "mmd_face": any(m["name"].startswith(prefix) for prefix in MMD_FACE_MATERIAL_PREFIXES),
                # ... and the wider PAINTED set: everything drawn from its albedo, which is the face block
                # plus the ear/head-wear materials above. A separate flag because the nose mark belongs to
                # the face alone, while "no lighting" applies to all of them.
                "mmd_unlit": any(m["name"].startswith(prefix) for prefix in MMD_UNLIT_MATERIAL_PREFIXES),
                "mmd_toon_name": model["textures"][m["toon_index"]] if m["toon_flag"] == 0 and 0 <= m["toon_index"] < len(model["textures"]) else None,
            },
        }
        if m["texture_index"] in pmx_to_gltf_image:
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": pmx_to_gltf_image[m["texture_index"]]}
        if d[3] < 1.0:
            mat["alphaMode"] = "BLEND"
        gltf_materials.append(mat)

    # ---- primitives: one per material, over its contiguous index range ----
    #
    # EACH PRIMITIVE GETS ITS OWN INDEX ACCESSOR over the shared index buffer view, because that is the
    # only place a glTF primitive can say WHICH triangles are its own: it has no firstIndex, so the range
    # IS the accessor's byteOffset + count. Handing every primitive the one accessor that covers the whole
    # buffer - which is what this did - makes all of them draw the entire mesh: the depth test then decides
    # which material is seen, and since the primitives are drawn in order the FIRST material wins. Measured
    # with the Khronos validator's cousin, a UV-region report over the exported GLB: 22 primitives, each
    # with 40944 triangles (the whole mesh) and each with the full-atlas UV bounding box, so the whole model
    # wore the face atlas - the body pale, the tie brown (the face atlas's fabric patch), the "hair" showing
    # that atlas's dark and mint regions.
    # THE FACE BLOCK'S OWN PLANE, accumulated here because this is where the per-material index
    # ranges and the vertex normals meet: the average of a face's own normals IS the plane it lies
    # in, which is what the engine's face flattening needs. A guessed world axis measured wrong, and
    # taking the direction from the camera made the shading follow the viewer.
    face_normal_sum = [0.0, 0.0, 0.0]
    primitives = []
    start = 0
    for mi, m in enumerate(model["materials"]):
        count = m["face_count"]
        if count == 0:
            continue
        if m["name"].startswith(MMD_FACE_MATERIAL_PREFIXES):
            for vertex_index in model["indices"][start : start + count]:
                normal = model["normals"][vertex_index]
                face_normal_sum[0] += normal[0]
                face_normal_sum[1] += normal[1]
                face_normal_sum[2] += normal[2]
        index_acc_i = {
            "bufferView": index_view,
            "byteOffset": start * 4,  # UNSIGNED_INT indices, so the byte offset is 4 per index
            "componentType": 5125,
            "count": count,
            "type": "SCALAR",
        }
        accessors.append(index_acc_i)
        primitives.append(
            {
                "attributes": dict(
                    {"POSITION": pos_acc, "NORMAL": nrm_acc, "TEXCOORD_0": uv_acc, "_EDGESCALE": edge_acc},
                    **{f"TEXCOORD_{i + 1}": a for i, a in enumerate(extra_acc)},
                ),
                "indices": len(accessors) - 1,
                "material": mi,
                "extras": {"mmd_index_begin": start, "mmd_index_count": count},
            }
        )
        start += count
    if start != len(model["indices"]):
        raise SystemExit(f"material face counts cover {start} of {len(model['indices'])} indices")

    # ... and the averaged direction goes into the FACE materials' extras, which is the only place a
    # loader can read application data from (see the loader's extras callback). Written on every face
    # material rather than one of them, so no consumer has to know which material carries it.
    face_length = math.sqrt(sum(component * component for component in face_normal_sum))
    if face_length > 1e-6:
        face_normal = [round(component / face_length, 6) for component in face_normal_sum]
        for material in gltf_materials:
            if material["extras"]["mmd_face"]:
                material["extras"]["mmd_face_normal"] = face_normal
        print(f"  the face block faces {face_normal} (averaged over its own vertex normals)")
    else:
        print("  the face block has no usable normals; the engine keeps its own default direction")

    gltf = {
        "asset": {"version": "2.0", "generator": "pmx_to_glb.py (project tool)"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": model["name"]}],
        "meshes": [{"name": model["name"], "primitives": primitives}],
        "materials": gltf_materials,
        "textures": gltf_textures,
        "images": gltf_images,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
        "extras": {"pmx_version": model["version"], "pmx_uv_count": model["uv_count"]},
    }

    while len(blob) % 4:
        blob.append(0)
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode()
    while len(json_bytes) % 4:
        json_bytes += b" "
    total = 12 + 8 + len(json_bytes) + 8 + len(blob)
    with out_path.open("wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(blob)
    return {"primitives": len(primitives), "materials": len(gltf_materials), "images": len(gltf_images), "bytes": total}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    model = read_pmx(src)
    model["_dir"] = str(src.parent)
    print(f"PMX {model['version']}: '{model['name']}'  vertices={len(model['positions'])}  indices={len(model['indices'])}")
    print(f"  uv sets={model['uv_count']}  textures={len(model['textures'])}  materials={len(model['materials'])}")
    stats = write_glb(dst, model)
    print(f"wrote {dst}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())