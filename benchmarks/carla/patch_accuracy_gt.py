from pathlib import Path
import hashlib

EXPECTED_SHA256 = "dc5d8d68ae255b1bd4cd5359f26008b4b7688ac11a2f8b06d56b0f282b4dbf89"

p = Path.home() / "Desktop" / "KV260_Benchmark" / "carla" / "carla_kv260_accuracy_benchmark.py"
backup = Path(str(p) + ".before_gt")

if not p.exists():
    raise SystemExit("ERROR: benchmark source not found: " + str(p))

raw = p.read_bytes()
current_sha = hashlib.sha256(raw).hexdigest()

s = p.read_text(encoding="utf-8")

if "GT_CSV = os.path.join" in s and "collect_ground_truth" in s:
    raise SystemExit("ERROR: benchmark already appears to be patched; no changes made")

if current_sha != EXPECTED_SHA256:
    raise SystemExit(
        "ERROR: benchmark SHA256 is not the proven original copy.\n"
        "Expected: " + EXPECTED_SHA256 + "\n"
        "Found:    " + current_sha + "\n"
        "No changes made."
    )

if not backup.exists():
    backup.write_bytes(raw)


def replace_once(old, new, label):
    global s
    if old not in s:
        raise SystemExit("ERROR: patch point not found: " + label + "; no file written")
    s = s.replace(old, new, 1)


replace_once(
    "import os\nimport queue\n",
    "import os\nimport csv\nimport queue\n",
    "csv import"
)

replace_once(
    "DRAW_INTERSECTION_GT = True\n",
    """DRAW_INTERSECTION_GT = False  # keep accuracy images free of CARLA GT overlays

# Object-detection accuracy benchmark ground truth.
GT_MAX_DISTANCE_M = 60.0
GT_MIN_BOX_W = 4
GT_MIN_BOX_H = 4
GT_DIR = os.path.join(
    os.path.expanduser("~"),
    "Desktop", "KV260_Benchmark", "results"
)
GT_CSV = os.path.join(GT_DIR, "carla_ground_truth_run_01.csv")
GT_FRAMES_CSV = os.path.join(GT_DIR, "carla_frames_run_01.csv")
""",
    "GT constants"
)

helper = r'''
def coco_category_for_actor(actor):
    # Map benchmark CARLA actors to the COCO IDs used by the KV260 model.
    tid = actor.type_id.lower()

    if tid.startswith("walker.pedestrian."):
        return 0, "person"

    if not tid.startswith("vehicle."):
        return None

    if any(x in tid for x in ("diamondback", "gazelle", "crossbike")):
        return 1, "bicycle"

    if any(x in tid for x in ("harley", "kawasaki", "yamaha")):
        return 3, "motorbike"

    if any(x in tid for x in ("bus", "fusorosa")):
        return 5, "bus"

    if any(x in tid for x in ("carlacola", "cybertruck")):
        return 7, "truck"

    # Remaining four-wheel NPCs are evaluated as COCO car.
    return 2, "car"


def camera_calibration(width, height, fov_deg):
    k = np.identity(3, dtype=np.float64)
    focal = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    return k


def project_world_point(location, world_to_camera, k):
    point = np.array(
        [location.x, location.y, location.z, 1.0],
        dtype=np.float64)

    p = np.dot(world_to_camera, point)

    # CARLA/Unreal coordinates -> conventional camera coordinates.
    cam = np.array([p[1], -p[2], p[0]], dtype=np.float64)
    depth = cam[2]
    if depth <= 0.10:
        return None

    uvw = np.dot(k, cam)
    return (
        float(uvw[0] / uvw[2]),
        float(uvw[1] / uvw[2]),
        float(depth)
    )


def actor_2d_box(actor, camera_tf, world_to_camera, k):
    try:
        if not actor.is_alive:
            return None

        actor_tf = actor.get_transform()
        camera_loc = camera_tf.location
        actor_loc = actor_tf.location

        dx = actor_loc.x - camera_loc.x
        dy = actor_loc.y - camera_loc.y
        dz = actor_loc.z - camera_loc.z
        distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        if distance > GT_MAX_DISTANCE_M:
            return None

        # Reject actors whose origin is behind the camera.
        forward = camera_tf.get_forward_vector()
        if dx * forward.x + dy * forward.y + dz * forward.z <= 0.0:
            return None

        vertices = actor.bounding_box.get_world_vertices(actor_tf)
        projected = []
        for vertex in vertices:
            q = project_world_point(vertex, world_to_camera, k)
            if q is not None:
                projected.append(q)

        if len(projected) < 4:
            return None

        xs = [q[0] for q in projected]
        ys = [q[1] for q in projected]

        x1 = max(0.0, min(float(CAM_W - 1), min(xs)))
        y1 = max(0.0, min(float(CAM_H - 1), min(ys)))
        x2 = max(0.0, min(float(CAM_W - 1), max(xs)))
        y2 = max(0.0, min(float(CAM_H - 1), max(ys)))

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < GT_MIN_BOX_W or (y2 - y1) < GT_MIN_BOX_H:
            return None

        return x1, y1, x2, y2, distance

    except Exception:
        return None


def collect_ground_truth(camera, actors, k):
    camera_tf = camera.get_transform()
    world_to_camera = np.array(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64)

    gt = []
    for actor in actors:
        category = coco_category_for_actor(actor)
        if category is None:
            continue

        box = actor_2d_box(actor, camera_tf, world_to_camera, k)
        if box is None:
            continue

        class_id, class_name = category
        x1, y1, x2, y2, distance = box

        gt.append((
            class_id,
            class_name,
            x1, y1, x2, y2,
            distance,
            actor.id,
            actor.type_id
        ))

    return gt


'''

replace_once(
    "def spawn_lead_vehicle(world, carla_map, bp_lib, ego):\n",
    helper + "def spawn_lead_vehicle(world, carla_map, bp_lib, ego):\n",
    "GT helper insertion"
)

replace_once(
    """    walker_controllers = []
    frame_q = queue.Queue(maxsize=4)
""",
    """    walker_controllers = []
    frame_q = queue.Queue(maxsize=4)
    gt_file = None
    gt_writer = None
    frame_file = None
    frame_writer = None
""",
    "main GT variables"
)

replace_once(
    """        camera.listen(on_image)
        print("RGB camera attached")

        for _ in range(5):
""",
    """        camera.listen(on_image)
        print("RGB camera attached")

        os.makedirs(GT_DIR, exist_ok=True)

        gt_file = open(GT_CSV, "w", newline="")
        gt_writer = csv.writer(gt_file)
        gt_writer.writerow([
            "sent_frame_id", "carla_frame_id",
            "class_id", "class_name",
            "x1", "y1", "x2", "y2",
            "distance_m", "actor_id", "carla_type_id"
        ])

        frame_file = open(GT_FRAMES_CSV, "w", newline="")
        frame_writer = csv.writer(frame_file)
        frame_writer.writerow([
            "sent_frame_id", "carla_frame_id",
            "carla_timestamp_s", "gt_object_count"
        ])

        camera_k = camera_calibration(CAM_W, CAM_H, 90.0)

        print("GT CSV       :", GT_CSV)
        print("Frame CSV    :", GT_FRAMES_CSV)

        for _ in range(5):
""",
    "open GT CSVs"
)

replace_once(
    """            frame = image_to_bgr(image)
            jdist = junction_distance_ahead(carla_map, ego)
            draw_intersection_gt(frame, jdist)

            ok, enc = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            payload = enc.tobytes()
            sock.sendall(struct.pack("!I", len(payload)))
            sock.sendall(payload)
            sent += 1
""",
    """            frame = image_to_bgr(image)

            gt_actors = []
            gt_actors.extend(npc_vehicles)
            gt_actors.extend(walkers)
            gt_objects = collect_ground_truth(camera, gt_actors, camera_k)

            jdist = junction_distance_ahead(carla_map, ego)
            draw_intersection_gt(frame, jdist)

            ok, enc = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            payload = enc.tobytes()
            sock.sendall(struct.pack("!I", len(payload)))
            sock.sendall(payload)
            sent += 1

            # Log only after this JPEG has been successfully transmitted.
            sent_frame_id = sent

            frame_writer.writerow([
                sent_frame_id,
                image.frame,
                "{:.6f}".format(image.timestamp),
                len(gt_objects)
            ])

            for obj in gt_objects:
                (class_id, class_name,
                 x1, y1, x2, y2,
                 distance_m, actor_id, type_id) = obj

                gt_writer.writerow([
                    sent_frame_id,
                    image.frame,
                    class_id,
                    class_name,
                    "{:.2f}".format(x1),
                    "{:.2f}".format(y1),
                    "{:.2f}".format(x2),
                    "{:.2f}".format(y2),
                    "{:.3f}".format(distance_m),
                    actor_id,
                    type_id
                ])

            if sent % 30 == 0:
                gt_file.flush()
                frame_file.flush()
""",
    "per-frame GT logging"
)

replace_once(
    """    finally:
        print("\\nCleaning up...")
        if sock is not None:
""",
    """    finally:
        print("\\nCleaning up...")

        if gt_file is not None:
            try:
                gt_file.flush()
                gt_file.close()
                print("Ground truth saved:", GT_CSV)
            except Exception:
                pass

        if frame_file is not None:
            try:
                frame_file.flush()
                frame_file.close()
                print("Frame manifest saved:", GT_FRAMES_CSV)
            except Exception:
                pass

        if sock is not None:
""",
    "GT cleanup"
)

# Only write after every expected patch point was found successfully.
p.write_text(s, encoding="utf-8")

new_sha = hashlib.sha256(p.read_bytes()).hexdigest()
print("Accuracy benchmark GT patch applied successfully.")
print("Modified ONLY:", p)
print("Backup:", backup)
print("New SHA256:", new_sha)
print("GT CSV:", Path.home() / "Desktop" / "KV260_Benchmark" / "results" / "carla_ground_truth_run_01.csv")
print("Frame CSV:", Path.home() / "Desktop" / "KV260_Benchmark" / "results" / "carla_frames_run_01.csv")
