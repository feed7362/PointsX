"""Render one body: front + side RGB, clothed mask and nude-body mask. Runs inside Blender.

The old generator rendered the *nude* silhouette in mask mode, so the data taught a model that
silhouette == body — the opposite of the real failure, where clothing inflates every width. Here the
clothed render and the body mask come out of the same frame, so a model can learn the mapping the
product actually needs: clothed pixels -> body outline.

Camera is phone-like and jittered per render, matching the capture guidance: 26-28 mm equivalent,
1.8-3 m away, lens 0.9-1.4 m above the floor, small tilt.

usage (not importable — needs bpy):
    blender -b -P scripts/synthetic/blender_render.py -- --body runs/synthetic/bodies/f00000.obj \
        --out runs/synthetic/renders --seed 0 [--garment path.obj] [--samples 32]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy  # noqa: F401  (provided by Blender)
from mathutils import Vector

RES = 1024
VIEWS = {"front": 0.0, "side": math.pi / 2}


def argv_after_dashdash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    # EEVEE was renamed between Blender versions; take whatever this build offers.
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = next((e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES")
                                if e in engines), scene.render.engine)


def import_obj(path: Path):
    # Anny meshes are already Z-up; the importer's default Y-up conversion lays them on their side.
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = path.stem
    return obj


def body_dimensions(obj):
    zs = [(obj.matrix_world @ v.co).z for v in obj.data.vertices]
    return min(zs), max(zs)


def place_camera(z_floor: float, z_top: float, angle: float, rng: random.Random):
    """Phone-like camera: 26-28 mm equivalent, lens 0.9-1.4 m up, +-4 deg tilt, standing back far
    enough that the body fills ~`fill` of the frame — what the capture guidance asks users for."""
    focal_mm = rng.uniform(26.0, 28.0)
    lens_h = rng.uniform(0.9, 1.4)
    fill = rng.uniform(0.80, 0.92)
    half_fov = math.atan(36.0 / 2.0 / focal_mm)          # square sensor crop: same both axes
    dist = (z_top - z_floor) / (2.0 * fill * math.tan(half_fov))
    tilt = math.radians(rng.uniform(-4.0, 4.0))
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = focal_mm
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector((math.sin(angle) * dist, -math.cos(angle) * dist, z_floor + lens_h))
    target_z = z_floor + (z_top - z_floor) * 0.55
    direction = Vector((0.0, 0.0, target_z)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.rotation_euler.x += tilt
    bpy.context.scene.camera = cam
    return {"focal_mm": round(focal_mm, 2), "distance_m": round(dist, 3),
            "lens_height_m": round(lens_h, 3), "tilt_deg": round(math.degrees(tilt), 2)}


def add_light(rng: random.Random, hdri: Path | None):
    world = bpy.data.worlds.new("w")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    if hdri and hdri.is_file():
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(hdri))
        links.new(env.outputs["Color"], bg.inputs["Color"])
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
        tex = nodes.new("ShaderNodeTexCoord")
        links.new(tex.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    else:
        bg.inputs["Color"].default_value = (0.55, 0.55, 0.58, 1.0)
    bg.inputs["Strength"].default_value = rng.uniform(0.7, 1.4)


def flat_material(rgba):
    mat = bpy.data.materials.new("flat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def render_to(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, type=Path)
    ap.add_argument("--garment", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hdri-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args(argv_after_dashdash())
    # Blender's working directory is its own install dir, so relative paths from the caller break.
    for name in ("body", "garment", "out", "hdri_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve())
    rng = random.Random(args.seed)

    meta = {"body": args.body.name, "garment": args.garment.name if args.garment else None,
            "seed": args.seed, "views": {}}
    hdris = sorted(args.hdri_dir.glob("*.hdr")) if args.hdri_dir else []
    hdri = rng.choice(hdris) if hdris else None

    for view, angle in VIEWS.items():
        reset_scene()
        if hasattr(bpy.context.scene, "eevee"):
            bpy.context.scene.eevee.taa_render_samples = args.samples
        body = import_obj(args.body)
        z0, z1 = body_dimensions(body)
        garment = import_obj(args.garment) if args.garment else None
        add_light(rng, hdri)
        cam = place_camera(z0, z1, angle, random.Random(args.seed))      # same camera for both passes
        meta["views"][view] = cam

        # 1. dressed photo
        render_to(args.out / f"{args.body.stem}_{view}_rgb.png")

        # 2. clothed silhouette: everything white on transparent
        white = flat_material((1, 1, 1, 1))
        for o in [body] + ([garment] if garment else []):
            o.data.materials.clear(); o.data.materials.append(white)
        bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        render_to(args.out / f"{args.body.stem}_{view}_mask_clothed.png")

        # 3. body silhouette: the same frame with the garment hidden — this pair is the whole point
        if garment:
            garment.hide_render = True
        render_to(args.out / f"{args.body.stem}_{view}_mask_body.png")

    (args.out / f"{args.body.stem}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[render] {args.body.stem}: {len(VIEWS) * 3} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
