#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA 0.9.9.2 -> Ethernet -> KV260 LIVE SENDER V2

Adds:
- extra NPC vehicles and pedestrians so the KV260 YOLOv5 Nano has targets
- ego spawn chosen near a junction when possible
- CARLA map-ground-truth intersection indicator overlaid on the camera frame

Important:
- Car/person detection is still performed by the KV260 DPU YOLO model.
- Intersection status is NOT a YOLO class here; it comes from CARLA map data
  and is explicitly labeled [CARLA MAP GT].
- TCP framing is unchanged: 4-byte big-endian JPEG length + JPEG bytes.
"""

import glob
import os
import queue
import random
import socket
import struct
import sys
import time

CARLA_ROOT = r"C:\Users\Md Mostafizur Rahman\Downloads\CARLA_0.9.9.2\WindowsNoEditor"
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
TM_PORT = 8000

KV260_IP = "192.168.50.2"
KV260_PORT = 5000

CAM_W = 640
CAM_H = 480
CAM_FPS = 10
JPEG_QUALITY = 85

NUM_NPC_VEHICLES = 12
NUM_PEDESTRIANS = 12
INTERSECTION_LOOKAHEAD_M = 30
INTERSECTION_STEP_M = 5
DRAW_INTERSECTION_GT = True

# CARLA 0.9.9.2 Python 3.7 egg
EGG = os.path.join(
    CARLA_ROOT, "PythonAPI", "carla", "dist",
    "carla-0.9.9-py3.7-win-amd64.egg"
)
if not glob.glob(EGG):
    raise RuntimeError("CARLA egg not found: " + EGG)
sys.path.append(glob.glob(EGG)[0])

import carla
import cv2
import numpy as np


def set_autopilot_compat(vehicle, enabled=True):
    try:
        vehicle.set_autopilot(enabled, TM_PORT)
    except TypeError:
        vehicle.set_autopilot(enabled)


def is_junction(wp):
    if wp is None:
        return False
    if hasattr(wp, "is_junction"):
        try:
            return bool(wp.is_junction)
        except Exception:
            pass
    if hasattr(wp, "is_intersection"):
        try:
            return bool(wp.is_intersection)
        except Exception:
            pass
    return False


def waypoint_for(carla_map, location):
    try:
        return carla_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except Exception:
        return carla_map.get_waypoint(location)


def junction_distance_ahead(carla_map, ego):
    wp = waypoint_for(carla_map, ego.get_location())
    if wp is None:
        return None
    if is_junction(wp):
        return 0

    frontier = [wp]
    distance = 0
    while distance < INTERSECTION_LOOKAHEAD_M and frontier:
        distance += INTERSECTION_STEP_M
        nxt_frontier = []
        for cur in frontier[:12]:
            try:
                nxts = cur.next(float(INTERSECTION_STEP_M))
            except Exception:
                nxts = []
            for nxt in nxts:
                if is_junction(nxt):
                    return distance
                nxt_frontier.append(nxt)
        frontier = nxt_frontier[:12]
    return None


def choose_spawn_near_junction(carla_map, spawn_points):
    pts = list(spawn_points)
    random.shuffle(pts)
    for sp in pts:
        wp = waypoint_for(carla_map, sp.location)
        if wp is None:
            continue
        if is_junction(wp):
            return sp
        frontier = [wp]
        for _ in range(5):  # up to ~25 m
            nxt_frontier = []
            for cur in frontier[:8]:
                try:
                    nxts = cur.next(5.0)
                except Exception:
                    nxts = []
                for nxt in nxts:
                    if is_junction(nxt):
                        return sp
                    nxt_frontier.append(nxt)
            frontier = nxt_frontier[:8]
    return random.choice(spawn_points)


def draw_intersection_gt(frame, distance):
    if not DRAW_INTERSECTION_GT:
        return
    h, w = frame.shape[:2]
    if distance == 0:
        text = "INTERSECTION NOW  [CARLA MAP GT]"
        bg = (0, 0, 180)
    elif distance is not None:
        text = "INTERSECTION AHEAD: {} m  [CARLA MAP GT]".format(int(distance))
        bg = (0, 120, 220)
    else:
        text = "ROAD SEGMENT  [CARLA MAP GT]"
        bg = (50, 50, 50)

    y1 = h - 34
    cv2.rectangle(frame, (0, y1), (w, h), bg, -1)
    cv2.putText(frame, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


def image_to_bgr(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def spawn_lead_vehicle(world, carla_map, bp_lib, ego):
    """Try to guarantee one clear car target in front of ego."""
    try:
        wp = waypoint_for(carla_map, ego.get_location())
        ahead = wp.next(18.0)
        if not ahead:
            return None
        tf = ahead[0].transform
        tf.location.z += 0.3
        bps = list(bp_lib.filter("vehicle.*"))
        random.shuffle(bps)
        for bp in bps[:12]:
            a = world.try_spawn_actor(bp, tf)
            if a is not None:
                set_autopilot_compat(a, True)
                return a
    except Exception:
        pass
    return None


def spawn_npc_vehicles(world, bp_lib, spawn_points, ego, count):
    actors = []
    bps = list(bp_lib.filter("vehicle.*"))
    pts = list(spawn_points)
    random.shuffle(pts)
    ego_loc = ego.get_location()

    for sp in pts:
        if len(actors) >= count:
            break
        if sp.location.distance(ego_loc) < 8.0:
            continue
        a = world.try_spawn_actor(random.choice(bps), sp)
        if a is None:
            continue
        try:
            set_autopilot_compat(a, True)
            actors.append(a)
        except Exception:
            a.destroy()
    return actors


def spawn_pedestrians(world, bp_lib, count):
    walkers = []
    controllers = []
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    controller_bp = bp_lib.find("controller.ai.walker")

    attempts = 0
    while len(walkers) < count and attempts < count * 8:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
        if walker is None:
            continue
        try:
            controller = world.spawn_actor(
                controller_bp, carla.Transform(), attach_to=walker
            )
        except Exception:
            walker.destroy()
            continue
        walkers.append(walker)
        controllers.append(controller)

    try:
        world.tick()
    except Exception:
        pass

    for c in controllers:
        try:
            c.start()
            dst = world.get_random_location_from_navigation()
            if dst is not None:
                c.go_to_location(dst)
            c.set_max_speed(random.uniform(1.0, 1.7))
        except Exception:
            pass

    return walkers, controllers


def main():
    world = None
    tm = None
    original_settings = None
    ego = None
    camera = None
    sock = None
    npc_vehicles = []
    walkers = []
    walker_controllers = []
    frame_q = queue.Queue(maxsize=4)

    print("=" * 66)
    print(" CARLA 0.9.9.2 -> Ethernet -> KV260 LIVE SENDER V2")
    print("=" * 66)
    print("KV260        : {}:{}".format(KV260_IP, KV260_PORT))
    print("Camera       : {}x{} @ {} FPS".format(CAM_W, CAM_H, CAM_FPS))
    print("JPEG quality : {}".format(JPEG_QUALITY))
    print("NPC vehicles : {}".format(NUM_NPC_VEHICLES))
    print("Pedestrians  : {}".format(NUM_PEDESTRIANS))
    print("Intersection : CARLA map GT, {} m look-ahead".format(
        INTERSECTION_LOOKAHEAD_M))
    print("=" * 66)

    try:
        client = carla.Client(CARLA_HOST, CARLA_PORT)
        client.set_timeout(20.0)
        world = client.get_world()
        carla_map = world.get_map()
        bp_lib = world.get_blueprint_library()

        print("Connected to CARLA")
        print("Map:", carla_map.name.split("/")[-1])

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(CAM_FPS)
        world.apply_settings(settings)

        tm = client.get_trafficmanager(TM_PORT)
        try:
            tm.set_synchronous_mode(True)
            tm.set_global_distance_to_leading_vehicle(2.5)
        except Exception:
            pass

        spawn_points = carla_map.get_spawn_points()
        ego_bp = bp_lib.find("vehicle.mercedes-benz.coupe")
        preferred = choose_spawn_near_junction(carla_map, spawn_points)
        ordered = [preferred] + [
            sp for sp in spawn_points
            if sp.location.distance(preferred.location) > 1.0
        ]
        for sp in ordered:
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError("Could not spawn ego vehicle")
        set_autopilot_compat(ego, True)
        print("Ego vehicle :", ego.type_id)
        print("Autopilot   : ON")

        lead = spawn_lead_vehicle(world, carla_map, bp_lib, ego)
        if lead is not None:
            npc_vehicles.append(lead)
            print("Lead vehicle: spawned for early YOLO car target")

        npc_vehicles.extend(spawn_npc_vehicles(
            world, bp_lib, spawn_points, ego,
            max(0, NUM_NPC_VEHICLES - len(npc_vehicles))))
        print("NPC vehicles spawned:", len(npc_vehicles))

        walkers, walker_controllers = spawn_pedestrians(
            world, bp_lib, NUM_PEDESTRIANS)
        print("Pedestrians spawned:", len(walkers))

        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAM_W))
        camera_bp.set_attribute("image_size_y", str(CAM_H))
        camera_bp.set_attribute("fov", "90")
        try:
            camera_bp.set_attribute("sensor_tick", str(1.0 / float(CAM_FPS)))
        except Exception:
            pass

        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=1.7),
                            carla.Rotation(pitch=-5.0)),
            attach_to=ego)

        def on_image(image):
            try:
                frame_q.put_nowait(image)
            except queue.Full:
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_q.put_nowait(image)
                except queue.Full:
                    pass

        camera.listen(on_image)
        print("RGB camera attached")

        for _ in range(5):
            world.tick()

        print("Connecting to KV260...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect((KV260_IP, KV260_PORT))
        sock.settimeout(None)
        print("Connected to KV260")
        print()
        print("LIVE streaming started. Press Ctrl+C to stop.")
        print("Car/person boxes: KV260 YOLO DPU")
        print("Intersection label: CARLA map ground truth")
        print()

        sent = 0
        t0 = time.time()

        while True:
            world.tick()
            try:
                image = frame_q.get(timeout=2.0)
            except queue.Empty:
                print("WARNING: camera timeout")
                continue

            frame = image_to_bgr(image)
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

            if sent % 30 == 0:
                fps = sent / max(time.time() - t0, 1e-6)
                if jdist == 0:
                    junction = "NOW"
                elif jdist is not None:
                    junction = "{}m".format(int(jdist))
                else:
                    junction = "none"
                print("Sent {:6d} frames | sender {:5.2f} FPS | JPEG {:5.1f} KB | junction {}".format(
                    sent, fps, len(payload) / 1024.0, junction))

    except KeyboardInterrupt:
        print("\nCtrl+C received - stopping live sender...")
    except Exception as e:
        print("\nERROR: {}: {}".format(type(e).__name__, e))
    finally:
        print("\nCleaning up...")
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.destroy()
            except Exception:
                pass
        for c in walker_controllers:
            try:
                c.stop()
            except Exception:
                pass
        for c in walker_controllers:
            try:
                c.destroy()
            except Exception:
                pass
        for w in walkers:
            try:
                w.destroy()
            except Exception:
                pass
        for a in npc_vehicles:
            try:
                a.destroy()
            except Exception:
                pass
        if ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass
        if tm is not None:
            try:
                tm.set_synchronous_mode(False)
            except Exception:
                pass
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
