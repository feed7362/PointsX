"""Blender headless rendering script for synthetic body dataset generation.

Run via:
    blender --background --python blender_render.py -- \
        --manifest path/to/manifest.json \
        --out-dir   path/to/output \
        --assets    path/to/assets

The script is imported by Blender's embedded Python, so it must NOT import
modules that are unavailable inside Blender (e.g. smplx, torch).
All SMPL-X data (OBJ files + landmark JSON) is pre-computed by the main
pipeline and passed to Blender via the manifest file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Тепер імпорт спрацює без помилок
# ─────────────────────────────────────────────────────────────────────────────
# Guard: only import bpy when running inside Blender
# ─────────────────────────────────────────────────────────────────────────────
try:
    import bpy
    import mathutils

    _IN_BLENDER = True
except ImportError:
    _IN_BLENDER = False

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W = 640
IMG_H = 640
FOCAL_LENGTH_MM = 50  # wide-angle to fit full body at close range
SENSOR_WIDTH_MM = 36.0
RENDER_ENGINE = "BLENDER_EEVEE"  # "CYCLES" (best on headless/Colab) or "BLENDER_EEVEE" (fast with display)
RENDER_SAMPLES = 16  # Cycles samples (minimal — flat shading doesn't need many bounces)
DENOISER = "OPENIMAGEDENOISE"

# Camera front-view and side-view base positions / rotations
# SMPL-X origin is at pelvis (~0m Z); head ~+0.85m, feet ~-0.85m.
# Close range (1.5-2.5m) with 28mm lens → body fills the frame.
CAMERAS = {
    "front": {
        "location": (0.0, -2.0, 0.0),
        "rotation": (math.radians(90), 0.0, 0.0),
    },
    "side": {
        "location": (2.0, 0.0, 0.0),
        "rotation": (math.radians(90), 0.0, math.radians(90)),
    },
}

# Camera random jitter ranges
CAM_DISTANCE_RANGE = (2.2, 2.8)  # metres (close, like real photos)
CAM_HEIGHT_RANGE = (0.8, 1.1)  # metres (roughly pelvis-level, slight variation)
CAM_HORIZ_JITTER = math.radians(2)  # subtle horizontal jitter
CAM_TILT_PROB = 0.15  # 15% chance of ±5° tilt
CAM_TILT_RANGE = math.radians(5)

# Skin tone names (Fitzpatrick I-VI × male/female = 12; 30 total = 5 per combo)
SKIN_TEXTURE_PATTERN = "skin_{:02d}.png"  # assets/textures/skin_01.png … skin_30.png
N_SKIN_TEXTURES = 30

# Clothing presets
TIGHT_CLOTHING_OBJS = [
    "clothing_leggings_top.obj",
    "clothing_swimsuit.obj",
    "clothing_boxers_singlet.obj",
    "clothing_biker_shorts.obj",
    "clothing_sports_set.obj",
]
BAGGY_CLOTHING_OBJS = [
    "clothing_hoodie_pants.obj",
    "clothing_dress.obj",
    "clothing_wide_jeans.obj",
]
TIGHT_PROB = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Blender scene setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear_scene() -> None:
    """Delete all mesh objects and lights from the current scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Also clear orphaned data
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        bpy.data.images.remove(block)


def setup_render_settings(use_gpu: bool = True, engine: str = RENDER_ENGINE) -> None:
    """Configure render engine (EEVEE or Cycles)."""
    scene = bpy.context.scene
    scene.render.resolution_x = IMG_W
    scene.render.resolution_y = IMG_H
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 85

    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        cycles = scene.cycles
        if use_gpu:
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            cycles.device = "GPU"
            cycles.samples = RENDER_SAMPLES
        else:
            cycles.device = "CPU"
            cycles.samples = max(32, RENDER_SAMPLES // 2)
        cycles.use_denoising = True
        cycles.denoiser = DENOISER
        # Minimize ray bounces — no need for accurate light transport
        cycles.max_bounces = 1
        cycles.diffuse_bounces = 1
        cycles.glossy_bounces = 0
        cycles.transmission_bounces = 0
        cycles.transparent_max_bounces = 0
        cycles.volume_bounces = 0
    else:
        # EEVEE — real-time rasterizer, 10-50x faster than Cycles
        scene.render.engine = "BLENDER_EEVEE"
        eevee = scene.eevee

        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 32

        # Підтримка нового EEVEE-Next (Blender 4.2+ / 5.1)
        if hasattr(eevee, "use_raytracing"):
            eevee.use_raytracing = True
        else:
            # Підтримка старого EEVEE (Blender 3.6 / 4.1 для Colab)
            if hasattr(eevee, "use_ssr"): eevee.use_ssr = True
            if hasattr(eevee, "use_gtao"): eevee.use_gtao = True
            if hasattr(eevee, "use_soft_shadows"): eevee.use_soft_shadows = True

    # Z-pass for occlusion detection (Cycles only; EEVEE uses projection-based)
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = False  # skip compositor for speed


def _setup_compositor(scene) -> None:
    """Add subtle film grain and colour grading via compositor."""
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    viewer = tree.nodes.new("CompositorNodeViewer")

    # Colour grade: slight contrast + saturation boost
    hue_sat = tree.nodes.new("CompositorNodeHueSat")
    hue_sat.inputs["Saturation"].default_value = 1.05
    hue_sat.inputs["Value"].default_value = 1.03

    # Film grain
    noise_tex = tree.nodes.new("CompositorNodeTexture")
    try:
        noise_tex.texture = bpy.data.textures.new("grain", type="CLOUDS")
        noise_tex.texture.noise_scale = 0.5
    except Exception:
        pass

    mix_grain = tree.nodes.new("CompositorNodeMixRGB")
    mix_grain.blend_type = "ADD"
    mix_grain.inputs["Fac"].default_value = 0.02

    # Link nodes
    links = tree.links
    links.new(render_layers.outputs["Image"], hue_sat.inputs["Image"])
    links.new(hue_sat.outputs["Image"], mix_grain.inputs[1])
    links.new(mix_grain.outputs["Image"], composite.inputs["Image"])
    links.new(mix_grain.outputs["Image"], viewer.inputs["Image"])


def load_hdri(hdri_path: str) -> None:
    """Set HDRI world lighting from a .hdr / .exr file."""
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    env = nodes.new("ShaderNodeTexEnvironment")
    mapping = nodes.new("ShaderNodeMapping")
    coord = nodes.new("ShaderNodeTexCoord")
    out = nodes.new("ShaderNodeOutputWorld")

    env.image = bpy.data.images.load(hdri_path)
    # Random HDRI rotation for variety
    mapping.inputs["Rotation"].default_value[2] = random.uniform(0, 2 * math.pi)

    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    links.new(env.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    bg.inputs["Strength"].default_value = random.uniform(0.8, 1.2)


def add_fill_light() -> None:
    """Add soft frontal fill light + subtle rim light."""
    # Fill light behind camera
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.5, 1.5))
    fill = bpy.context.active_object
    fill.data.energy = random.uniform(300, 800)
    fill.data.size = 2.0
    fill.rotation_euler = (math.radians(90), 0, 0)

    # Rim light from side/above
    bpy.ops.object.light_add(type="SPOT", location=(1.5, -1.0, 2.5))
    rim = bpy.context.active_object
    rim.data.energy = random.uniform(50, 200)
    rim.data.spot_size = math.radians(30)
    # Point toward subject origin
    direction = mathutils.Vector((0, 0, 0)) - mathutils.Vector(rim.location)
    rim.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def import_body_obj(obj_path: str) -> object:
    """Import body OBJ mesh; return the imported object."""
    bpy.ops.wm.obj_import(filepath=obj_path)
    # The imported object is now the active/selected one
    return bpy.context.selected_objects[0]


def _bsdf_input(bsdf, *names):
    """Look up a Principled BSDF input by name, trying aliases for cross-version compat.

    Blender 4.0+ renamed several inputs:
        "Subsurface" → "Subsurface Weight"
        "Specular"   → "Specular IOR Level"
    """
    for name in names:
        if name in bsdf.inputs:
            return bsdf.inputs[name]
    raise KeyError(f"BSDF input not found (tried {names})")


def apply_skin_material(body_obj, texture_path: str) -> None:
    """Apply photorealistic skin Principled BSDF with SSS to the body mesh."""
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")

    # Simple diffuse — no SSS, no specular (fast rendering for training data)
    bsdf.inputs["Roughness"].default_value = 1.0
    _bsdf_input(bsdf, "Specular IOR Level", "Specular").default_value = 0.0

    # UV texture
    if texture_path and Path(texture_path).exists():
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(texture_path)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        # Fallback: neutral skin tone
        bsdf.inputs["Base Color"].default_value = (0.8, 0.6, 0.5, 1.0)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Assign to mesh
    if body_obj.data.materials:
        body_obj.data.materials[0] = mat
    else:
        body_obj.data.materials.append(mat)


def import_clothing(clothing_path: str, body_obj) -> object | None:
    """Import clothing OBJ and shrinkwrap it to the body mesh."""
    if not Path(clothing_path).exists():
        return None

    bpy.ops.wm.obj_import(filepath=clothing_path)
    cloth_obj = bpy.context.selected_objects[0]

    mod = cloth_obj.modifiers.new("ShrinkWrap", "SHRINKWRAP")
    mod.target = body_obj
    mod.offset = 0.003
    mod.wrap_method = "NEAREST_SURFACEPOINT"
    bpy.context.view_layer.objects.active = cloth_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    _apply_fabric_material(cloth_obj)
    return cloth_obj


def _apply_fabric_material(obj) -> None:
    """Apply random-coloured PBR fabric shader to clothing mesh."""
    mat = bpy.data.materials.new("fabric")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out = nodes.new("ShaderNodeOutputMaterial")

    # Random hue, contrasting saturation
    hue = random.random()
    bsdf.inputs["Base Color"].default_value = (*_hsv_to_rgb(hue, 0.7, 0.6), 1.0)
    bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.6)

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB (all values in [0, 1])."""
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


# ─────────────────────────────────────────────────────────────────────────────
# Mask-mode helpers (white-on-black silhouette for YOLO-seg training)
# ─────────────────────────────────────────────────────────────────────────────

def apply_emission_material(obj, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
    """Apply a pure Emission shader — no lighting, no shadows, solid color."""
    mat = bpy.data.materials.new("emission")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    emit.inputs["Color"].default_value = color
    emit.inputs["Strength"].default_value = 1.0
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def setup_black_world() -> None:
    """Pure black world background (no HDRI, no ambient light)."""
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    out = nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def setup_camera(view: str, rng: random.Random) -> tuple[object, dict]:
    """Create and position a camera for the given view."""
    base = CAMERAS[view]
    loc = list(base["location"])

    # Random distance jitter (scale from origin)
    dist = rng.uniform(*CAM_DISTANCE_RANGE)
    current_dist = math.sqrt(loc[0] ** 2 + loc[1] ** 2)
    if current_dist > 0:
        scale = dist / current_dist
        loc[0] *= scale
        loc[1] *= scale

    # Camera height jitter
    loc[2] = rng.uniform(*CAM_HEIGHT_RANGE)

    bpy.ops.object.camera_add(location=loc)
    cam_obj = bpy.context.active_object

    # Направляємо камеру на центр тіла (Z = 0.9m)
    look_at_point = mathutils.Vector((0.0, 0.0, 0.9))
    direction = look_at_point - cam_obj.location
    rot_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Always apply ±5° horizontal jitter
    rot_euler.z += rng.uniform(-CAM_HORIZ_JITTER, CAM_HORIZ_JITTER)

    # 30% chance of ±10° tilt
    if rng.random() < CAM_TILT_PROB:
        rot_euler.x += rng.uniform(-CAM_TILT_RANGE, CAM_TILT_RANGE)

    cam_obj.rotation_euler = rot_euler
    bpy.context.scene.camera = cam_obj

    cam_obj.data.lens = FOCAL_LENGTH_MM
    cam_obj.data.sensor_width = SENSOR_WIDTH_MM
    cam_obj.data.dof.use_dof = False

    params = {
        "location": tuple(loc),
        "rotation": (rot_euler.x, rot_euler.y, rot_euler.z),
        "focal_length_mm": FOCAL_LENGTH_MM,
        "sensor_width_mm": SENSOR_WIDTH_MM,
    }
    return cam_obj, params


# ─────────────────────────────────────────────────────────────────────────────
# Z-buffer extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_z_buffer(render_path: str) -> "np.ndarray | None":
    """Extract Z-pass depth buffer, load it into Numpy, and delete the EXR file."""
    try:
        import numpy as np
        import os

        rp = Path(render_path)

        # Підтримка і старих версій (з 0001), і Blender 5.1+ (без номера)
        z_path_v4 = rp.parent / f"{rp.stem}_depth_0001.exr"
        z_path_v5 = rp.parent / f"{rp.stem}_depth_.exr"

        z_path = z_path_v5 if z_path_v5.exists() else z_path_v4

        if not z_path.exists():
            print(f"Warning: Z-buffer not found for {rp.stem}")
            return None

        img = bpy.data.images.load(str(z_path))
        pixels = np.array(img.pixels[:], dtype=np.float32)

        if pixels.size == 0:
            raise RuntimeError(f"Empty EXR pixel buffer: {z_path.name}")

        # Single-layer EXR can appear as 1 channel (H*W) or 4 channels (H*W*4).
        if pixels.size == IMG_H * IMG_W:
            z = pixels.reshape(IMG_H, IMG_W)
        elif pixels.size == IMG_H * IMG_W * 4:
            z = pixels.reshape(IMG_H, IMG_W, 4)[:, :, 0]
        else:
            raise RuntimeError(
                f"Unexpected EXR buffer size {pixels.size} for {IMG_W}x{IMG_H}"
            )

        # Blender stores image data bottom-up.
        z = z[::-1, :].copy().astype(np.float32)

        # Очищуємо пам'ять Blender
        bpy.data.images.remove(img)

        os.remove(z_path)

        return z
    except Exception as e:
        print(f"Z-buffer error: {e}")
        return None


def _is_occluded_raycast(scene, cam_obj, target_point, margin: float = 0.01) -> bool:
    """Return True if geometry blocks camera->target ray before the target point."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_origin = cam_obj.matrix_world.translation
    ray_vec = target_point - cam_origin
    target_dist = float(ray_vec.length)
    if target_dist <= margin:
        return False

    direction = ray_vec.normalized()
    max_dist = max(target_dist - margin, 0.0)
    hit, hit_loc, _normal, _face_idx, _obj, _matrix = scene.ray_cast(
        depsgraph, cam_origin, direction, distance=max_dist
    )
    if not hit:
        return False

    hit_dist = float((hit_loc - cam_origin).length)
    return hit_dist < (target_dist - margin)


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_sample(
        manifest_entry: dict,
        assets_dir: Path,
        out_dir: Path,
        use_gpu: bool = True,
        engine: str = RENDER_ENGINE,
        mode: str = "photo",
) -> dict:
    """Render all views for one body sample.

    Args:
        manifest_entry: dict with body_id, obj_path, landmarks_json_path, ...
        assets_dir:     Path to assets directory (HDRIs, textures, clothing)
        out_dir:        Root output directory
        use_gpu:        Use CUDA GPU if available
        engine:         "CYCLES" or "BLENDER_EEVEE"
        mode:           "photo" (full render with HDRI+textures+clothing) or
                        "mask"  (white-body emission on black bg, for YOLO-seg training)

    Returns:
        dict mapping view_name → {image_path, label_path, camera_params}
    """
    from pointsx.synthetic.annotator import (  # noqa: PLC0415
        classify_visibility,
        build_yolo_label,
        write_yolo_label,
    )
    from pointsx.synthetic.landmarks import select_pointsx16  # noqa: PLC0415
    import numpy as np
    import json as _json

    rng = random.Random(manifest_entry.get("seed", manifest_entry["body_id"]))

    # Load pre-computed landmarks (only used in photo mode for keypoint labels)
    if mode == "photo":
        landmarks_data = _json.loads(Path(manifest_entry["landmarks_json_path"]).read_text())
        selected_landmarks = select_pointsx16(landmarks_data["landmarks_3d"])
        landmarks_3d = [
            np.array(v, dtype=np.float32)
            for v in selected_landmarks
        ]
    else:
        landmarks_3d = None

    results = {}

    # ── Resume support: skip views that already have image (+ label for photo mode) ───
    body_id = manifest_entry["body_id"]
    views_todo = []
    for view in ("front", "side"):
        sample_id = f"s{body_id:05d}_{view}"
        # Check both train and val splits for existing output
        found = False
        for split in ("train", "val"):
            img = out_dir / split / "images" / f"{sample_id}.jpg"
            lbl = out_dir / split / "labels" / f"{sample_id}.txt"
            # Mask mode: polygon label is written by a separate script, so we only require image
            img_ok = img.exists()
            lbl_ok = lbl.exists() if mode == "photo" else True
            if img_ok and lbl_ok:
                results[view] = {"image_path": str(img), "label_path": str(lbl) if lbl_ok else None, "skipped": True}
                found = True
                break
        if not found:
            views_todo.append(view)

    if not views_todo:
        return results  # both views already done

    for view in views_todo:
        # ── Build scene ─────────────────────────────────────────────────
        clear_scene()
        setup_render_settings(use_gpu=use_gpu, engine=engine)

        if mode == "mask":
            # Mask mode: black background, white-emission body, no clothing, no HDRI/fill
            setup_black_world()
            body_obj = import_body_obj(manifest_entry["obj_path"])
            apply_emission_material(body_obj, color=(1.0, 1.0, 1.0, 1.0))
        else:
            # Photo mode: full photorealistic render
            hdri_dir = assets_dir / "hdri"
            hdri_files = sorted(hdri_dir.glob("*.hdr")) + sorted(hdri_dir.glob("*.exr"))
            if hdri_files:
                load_hdri(str(rng.choice(hdri_files)))
            add_fill_light()

            body_obj = import_body_obj(manifest_entry["obj_path"])

            tex_idx = manifest_entry.get("skin_texture_idx", rng.randint(1, N_SKIN_TEXTURES))
            tex_path = str(assets_dir / "textures" / SKIN_TEXTURE_PATTERN.format(tex_idx))
            apply_skin_material(body_obj, tex_path)

            cloth_dir = assets_dir / "clothing"
            if cloth_dir.exists():
                cloth_files = list(cloth_dir.glob("*.obj"))
                if cloth_files and rng.random() < 0.85:
                    cloth_path = str(rng.choice(cloth_files))
                    import_clothing(cloth_path, body_obj)

        # Camera (identical for photo and mask so masks align pixel-perfect)
        cam_obj, cam_params = setup_camera(view, rng)

        # ── Render ──────────────────────────────────────────────────────
        split = "train" if rng.random() > 0.20 else "val"
        sample_id = f"s{manifest_entry['body_id']:05d}_{view}"

        img_path = out_dir / split / "images" / f"{sample_id}.jpg"
        label_path = out_dir / split / "labels" / f"{sample_id}.txt"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Налаштування Композитора для Z-буфера ──
        scene = bpy.context.scene
        scene.render.filepath = str(img_path)

        # ── Налаштування Композитора (Підтримка Blender 4.x та 5.0+) ──

        try:
            if hasattr(scene, "compositing_node_group"):
                # ==========================================
                # НОВИЙ API: BLENDER 5.0+
                # ==========================================
                if scene.compositing_node_group is None:
                    tree = bpy.data.node_groups.new("PointsX_Comp", "CompositorNodeTree")
                    scene.compositing_node_group = tree
                else:
                    tree = scene.compositing_node_group

                tree.nodes.clear()
                tree.interface.clear()

                rlayers = tree.nodes.new(type="CompositorNodeRLayers")

                # У Blender 5.0+ CompositorNodeComposite замінено на NodeGroupOutput
                output = tree.nodes.new(type="NodeGroupOutput")
                tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
                tree.links.new(rlayers.outputs["Image"], output.inputs["Image"])

                if engine == "BLENDER_EEVEE":
                    # Use ray-cast occlusion below; depth EXR export is unnecessary.
                    pass

            else:
                # ==========================================
                # СТАРИЙ API: BLENDER 4.x ТА НИЖЧЕ
                # ==========================================
                scene.use_nodes = True
                tree = scene.node_tree
                tree.nodes.clear()

                rlayers = tree.nodes.new("CompositorNodeRLayers")
                composite = tree.nodes.new("CompositorNodeComposite")
                tree.links.new(rlayers.outputs["Image"], composite.inputs["Image"])

                if engine == "BLENDER_EEVEE":
                    # Use ray-cast occlusion below; depth EXR export is unnecessary.
                    pass

        except Exception as e:
            print(f"Failed to access compositor: {e}")
            raise

        # tree.nodes.clear()
        #
        # rlayers = tree.nodes.new("CompositorNodeRLayers")
        # composite = tree.nodes.new("CompositorNodeComposite")
        # tree.links.new(rlayers.outputs["Image"], composite.inputs["Image"])
        #
        # if engine == "CYCLES":
        #     file_out = tree.nodes.new("CompositorNodeOutputFile")
        #     file_out.format.file_format = "OPEN_EXR"
        #     file_out.format.color_depth = "32"
        #     file_out.base_path = str(img_path.parent)
        #     # Blender автоматично додасть номер кадру (0001) до цього імені
        #     file_out.file_slots[0].path = f"{img_path.stem}_depth_"
        #     tree.links.new(rlayers.outputs["Depth"], file_out.inputs[0])

        # Тепер рендер збереже і JPG, і EXR
        bpy.ops.render.render(write_still=True)

        if mode == "mask":
            # No keypoint labels for mask mode — polygons are generated post-hoc from the PNG
            results[view] = {
                "image_path": str(img_path),
                "label_path": None,
                "camera_params": cam_params,
            }
            continue

        # ── Annotate (photo mode only) ──────────────────────────────────
        import bpy_extras
        import mathutils

        # 1. Піднімаємо точки з підлоги (SMPL -> Blender)
        landmarks_blender = []
        for pt in landmarks_3d:
            blender_pt = [pt[0], -pt[2], pt[1]]
            landmarks_blender.append(blender_pt)

        coords_px_list = []
        depth_list = []

        scene = bpy.context.scene
        for pt in landmarks_blender:
            vec = mathutils.Vector(pt)

            proj = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, vec)

            px = proj.x * IMG_W
            py = (1.0 - proj.y) * IMG_H

            coords_px_list.append([px, py])
            depth_list.append(proj.z)

        coords_px = np.array(coords_px_list, dtype=np.float32)
        depth = np.array(depth_list, dtype=np.float32)

        # Visibility in image plane (off-frame vs candidate visible).
        # We intentionally avoid EXR depth parsing in Blender 5.1 path.
        visibility = classify_visibility(coords_px, depth, None, IMG_W, IMG_H)

        label = build_yolo_label(coords_px, visibility, IMG_W, IMG_H)
        if label:
            write_yolo_label(label, label_path)

        results[view] = {
            "image_path": str(img_path),
            "label_path": str(label_path) if label else None,
            "camera_params": cam_params,
            "n_visible_kp": int((visibility >= 1).sum()),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (called from Blender subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def main_blender() -> None:
    """Parse args and render all entries from the manifest file."""
    # Ensure pointsx is importable inside Blender's embedded Python.
    # Walk up from this file to find the src/ root: .../src/pointsx/synthetic/blender_render.py
    _src_dir = str(Path(__file__).resolve().parent.parent.parent)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    # Blender passes script args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender synthetic body renderer")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--out-dir", required=True, help="Output root directory")
    parser.add_argument("--assets", required=True, help="Path to assets directory")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Use GPU (CUDA) rendering")
    parser.add_argument("--engine", type=str, default=RENDER_ENGINE,
                        choices=["CYCLES", "BLENDER_EEVEE"],
                        help="Render engine (default: %(default)s)")
    parser.add_argument("--mode", type=str, default="photo",
                        choices=["photo", "mask"],
                        help="'photo' for full render, 'mask' for white-on-black silhouette")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First manifest entry index to process")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last manifest entry index (exclusive)")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    assets_dir = Path(args.assets)
    engine = args.engine
    mode = args.mode

    manifest = json.loads(manifest_path.read_text())
    entries = manifest[args.start_idx: args.end_idx]
    print(f"[blender_render] Mode: {mode} | Engine: {engine} | GPU: {args.gpu} | Bodies: {len(entries)}")

    # Results tracking
    results_path = out_dir / "render_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    n_ok, n_err = 0, 0

    with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Rendering"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            TextColumn("• [green]{task.fields[ok]} ok[/] [red]{task.fields[err]} err[/]"),
    ) as progress:
        task = progress.add_task("render", total=len(entries), ok=0, err=0)

        for entry in entries:
            try:
                res = render_sample(entry, assets_dir, out_dir, use_gpu=args.gpu, engine=engine, mode=mode)
                with open(results_path, "a") as f:
                    f.write(json.dumps({"body_id": entry["body_id"], **res}) + "\n")
                n_ok += 1
            except Exception as exc:
                progress.console.print(f"  [red]ERROR[/] body_id={entry['body_id']}: {exc}")
                import traceback
                traceback.print_exc(file=sys.stdout)
                n_err += 1

            progress.update(task, advance=1, ok=n_ok, err=n_err)

    print(f"[blender_render] Done. {n_ok} rendered, {n_err} errors.")


# ─────────────────────────────────────────────────────────────────────────────
# When running inside Blender
# ─────────────────────────────────────────────────────────────────────────────

if _IN_BLENDER:
    main_blender()
