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
# Share of the garment's height held in place at the top edge, so it hangs rather than falls off.
PIN_BAND = 0.04
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


def drape(garment, body, frames: int, stiffness: float) -> None:
    """Settle the garment shell onto the body with Blender's cloth solver.

    An offset shell alone is the shrink-wrap the 2026-07 review rejected: it follows every curve of
    the body, so a model trained on it would learn that the silhouette IS the body — the opposite of
    the real failure. Simulating gravity against the body as a collider is what makes a loose
    garment bridge the waist and hang off the hip, which is the effect we need in the data.
    """
    bpy.context.view_layer.objects.active = body
    body.modifiers.new("collision", type="COLLISION")
    body.collision.thickness_outer = 0.004

    # Pin the top edge — the shoulders of a top, the waistband of trousers. Without it gravity just
    # drags the shell off the body: the first run left the chest uncovered (1.000x over-read) and the
    # trousers heaped around the ankles (3.30x at the hip).
    zs = [(garment.matrix_world @ v.co).z for v in garment.data.vertices]
    z_hi, z_lo = max(zs), min(zs)
    pinned = [i for i, z in enumerate(zs) if z >= z_hi - PIN_BAND * (z_hi - z_lo)]
    group = garment.vertex_groups.new(name="pin")
    group.add(pinned, 1.0, "REPLACE")

    bpy.context.view_layer.objects.active = garment
    mod = garment.modifiers.new("cloth", type="CLOTH")
    st = mod.settings
    st.vertex_group_mass = "pin"
    st.quality = 5
    st.mass = 0.25
    st.tension_stiffness = stiffness
    st.compression_stiffness = stiffness
    st.shear_stiffness = stiffness * 0.5
    st.bending_stiffness = max(0.05, stiffness * 0.02)
    st.use_pressure = False
    mod.collision_settings.use_self_collision = False
    mod.collision_settings.distance_min = 0.004

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
    # Freeze the settled shape so the mask passes below see the same geometry as the RGB pass.
    bpy.context.view_layer.objects.active = garment
    bpy.ops.object.modifier_apply(modifier="cloth")


def render_to(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, type=Path)
    ap.add_argument("--garment", type=Path, default=None, action="append", dest="garments",
                    help="garment OBJ; repeat for several pieces (top, trousers)")
    ap.add_argument("--cloth-frames", type=int, default=0,
                    help="cloth simulation steps. DEFAULT OFF: the solver is unstable on these "
                         "garments — the trousers start intersecting the body at the crotch and the "
                         "collision response blows them into a 418 px billow where the geometry "
                         "alone gives two clean 112 px legs. The hanging-tube rule in "
                         "make_garment.py already bridges concavities, which is the property the "
                         "data needs; simulation would only add wrinkles. Fix the instability "
                         "(start the garment clear of the body, raise collision quality) before "
                         "turning this back on.")
    ap.add_argument("--stiffness", type=float, default=None,
                    help="fabric tension stiffness; default jitters 5-25 per render")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hdri-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args(argv_after_dashdash())
    # Blender's working directory is its own install dir, so relative paths from the caller break.
    for name in ("body", "out", "hdri_dir"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve())
    args.garments = [Path(g).resolve() for g in (args.garments or [])]
    rng = random.Random(args.seed)

    meta = {"body": args.body.name, "garments": [g.name for g in (args.garments or [])],
            "cloth_frames": args.cloth_frames, "seed": args.seed, "views": {}}
    hdris = sorted(args.hdri_dir.glob("*.hdr")) if args.hdri_dir else []
    hdri = rng.choice(hdris) if hdris else None

    for view, angle in VIEWS.items():
        reset_scene()
        if hasattr(bpy.context.scene, "eevee"):
            bpy.context.scene.eevee.taa_render_samples = args.samples
        body = import_obj(args.body)
        z0, z1 = body_dimensions(body)
        pieces = []
        for g in (args.garments or []):
            piece = import_obj(g)
            if args.cloth_frames > 0:
                stiffness = args.stiffness if args.stiffness is not None else rng.uniform(5.0, 25.0)
                drape(piece, body, args.cloth_frames, stiffness)
            cloth_mat = bpy.data.materials.new("cloth")
            cloth_mat.use_nodes = True
            bsdf = cloth_mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (*[rng.uniform(0.05, 0.6) for _ in range(3)], 1.0)
                if "Roughness" in bsdf.inputs:
                    bsdf.inputs["Roughness"].default_value = rng.uniform(0.6, 0.95)
            piece.data.materials.clear()
            piece.data.materials.append(cloth_mat)
            pieces.append(piece)
        add_light(rng, hdri)
        cam = place_camera(z0, z1, angle, random.Random(args.seed))      # same camera for both passes
        meta["views"][view] = cam

        # 1. dressed photo
        render_to(args.out / f"{args.body.stem}_{view}_rgb.png")

        # 2. clothed silhouette: everything white on transparent
        white = flat_material((1, 1, 1, 1))
        for o in [body, *pieces]:
            o.data.materials.clear(); o.data.materials.append(white)
        bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        render_to(args.out / f"{args.body.stem}_{view}_mask_clothed.png")

        # 3. body silhouette: the same frame with the garments hidden — this pair is the whole point
        for piece in pieces:
            piece.hide_render = True
        render_to(args.out / f"{args.body.stem}_{view}_mask_body.png")

    (args.out / f"{args.body.stem}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[render] {args.body.stem}: {len(VIEWS) * 3} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
