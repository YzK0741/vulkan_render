# README screenshots

These two images are the renderer's SHIPPED default configuration (traced global illumination with shaded
hits, SSAO, TAA, cascaded shadows, the IBL prefilter) at 1080x960, and they are CAPTURES rather than
hand-taken screenshots: each one is a fixed config plus a fixed frame count, so it can be regenerated and
compared instead of being replaced by whatever the next session's window happened to show.

They were produced with the instruments in `scripts/`, from a base config that is the generator's default
output with `[gui] show = false` (a debug overlay in a README image is noise) and `taa = true` (a still
frame that has accumulated its history, which is what the engine looks like in use):

    pwsh -File scripts/windows/capture.ps1 -Base snapshot_base.toml -Tag rc_helmet \
         -Model <repo>/gltf_model/DamagedHelmet.gltf -Camera "" -Frames 120
    pwsh -File scripts/windows/capture.ps1 -Base snapshot_base.toml -Tag rc_flight \
         -Model <assets>/FlightHelmet/glTF/FlightHelmet.gltf -Camera "" -Frames 120

`-Camera ""` is deliberate: it leaves the pose to the scene's own `camera_fit`, which for a static scene is
deterministic (the run prints the pose it resolved to). The captures land in `<BuildDir>/gi-probe/`.

The renderer's own PNG writer stores the frame essentially uncompressed (4.1 MB for a 1080x960 RGBA frame),
so the captures were re-encoded LOSSLESSLY with Pillow before being committed - the pixels are the
renderer's, byte for byte (checked with `ImageChops.difference`), and only the deflate stream differs. That
is a 13x reduction: 4,148,543 -> 322,611 bytes (DamagedHelmet) and -> 290,968 (FlightHelmet).
