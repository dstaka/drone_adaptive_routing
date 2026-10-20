"""Visual-only landmarks and deterministic world-anchored display contracts."""
import math
import numpy as np

PALETTE = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']


def priority_text(value, schema_version=2, decimals=2):
    """Format validated priorities without hiding fractional integer-profile inputs."""
    if schema_version == 3:
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError('Integer priority labels require an integer token in [1,100]')
        return str(value)
    return f'{value:.{decimals}f}'


def priority_color(value, priority_max=5):
    if (priority_max not in (5, 100) or not math.isfinite(value) or not 1 <= value <= priority_max
            or (priority_max == 100 and type(value) is not int)):
        raise ValueError('Priority outside the declared palette contract')
    if priority_max == 100:
        value = 1 + 4 * (value - 1) / 99
    lo = min(3, int(value) - 1)
    fraction = value - (lo + 1)
    colors = [[int(c[i:i+2], 16) for i in (1, 3, 5)] for c in PALETTE[lo:lo+2]]
    return '#' + ''.join(f'{int(math.floor(a + fraction * (b-a) + .5)):02x}' for a,b in zip(*colors))


def marker_style(priority, selected=False, visited=False, alert=False, priority_max=5):
    fill = priority_color(priority, priority_max)
    rgb = [int(fill[i:i+2],16)/255 for i in (1,3,5)]
    linear = [v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
    luminance = sum(w*v for w,v in zip([.2126,.7152,.0722],linear))
    return {'fill': fill, 'ink':'#ffffff' if luminance<.179 else '#111111',
            'symbols': [key for key, enabled in
            [('selected', selected), ('visited', visited), ('alert', alert)] if enabled]}


def scene_manifest(scenario):
    points = np.asarray(scenario.coords)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    # Near-top overview separates every row without hiding the drone in a large HUD.
    radius = max(float(np.ptp(points[:, 0])) * .5 / (1280/720), float(np.ptp(points[:, 1])) * .5, .6)
    height = (radius + .6) / math.tan(math.radians(40/2))
    camera = {'res': [1280, 720], 'pos': [float(center[0]), float(center[1] - .65), float(center[2] + height)],
              'lookat': center.tolist(), 'up': [0., 1., 0.], 'fov': 40., 'margin_m': .6}
    result = {'schema_version': 1, 'checkpoint_ids': scenario.checkpoint_ids, 'camera': camera,
            'rendering_method': 'neutral fixed Genesis landmarks with world-anchored priority badges',
            'priority_range': [1, 5], 'palette': PALETTE, 'fraction_mapping': 'linear RGB, round half up',
            'state_symbols': {'selected': 'double ring / 選択中', 'visited': 'check / 訪問済み', 'alert': '!'},
            'label_format': 'ID | 優先度 value', 'marker_radius_m': .045,
            'event_mapping': 'Apply events with simulation_step <= captured frame step, in event_seq order',
            'markers': [{'id': name, 'world_xyz': scenario.coords[i+1], 'visual_only': True,
                         'fixed': True, 'collision': False, 'priority': scenario.priorities[i]}
                        for i,name in enumerate(scenario.checkpoint_ids)]}
    if scenario.schema_version == 3:
        result.update(schema_version=2, priority_range=[1,100], priority_profile=scenario.priority_profile,
                      palette_version='integer100-linear-RGB-v1',
                      value_colors={str(p): priority_color(p, 100) for p in range(1,101)})
    return result


def project(point, camera):
    """Project with Genesis rasterizer's vertical field of view and camera basis."""
    pos, look = np.asarray(camera['pos']), np.asarray(camera['lookat'])
    forward = look-pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, camera['up'])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(point)-pos
    depth = np.dot(delta, forward)
    if depth <= 0:
        raise ValueError('Landmark is behind the camera')
    w,h = camera['res']
    focal = h / (2 * math.tan(math.radians(camera['fov']/2)))
    return [float(w/2 + focal*np.dot(delta,right)/depth), float(h/2 - focal*np.dot(delta,up)/depth)]


def install_scene(env, manifest):
    """Add fixed noncolliding landmarks before the pinned scene is built."""
    import genesis as gs
    env.mission_markers = []
    for marker in manifest['markers']:
        env.mission_markers.append(env.scene.add_entity(
            morph=gs.morphs.Mesh(file='meshes/sphere.obj', scale=manifest['marker_radius_m'],
                                 pos=tuple(marker['world_xyz']), fixed=True, collision=False),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(.7, .7, .7)))))
    camera = manifest['camera']
    env.cam = env.scene.add_camera(res=tuple(camera['res']), pos=tuple(camera['pos']),
                                  lookat=tuple(camera['lookat']), up=tuple(camera['up']),
                                  fov=camera['fov'], GUI=False)


def validate_scene(manifest):
    """Reject invalid camera geometry before allocating the physics runtime."""
    camera = manifest['camera']
    if (not np.isfinite([*camera['pos'], *camera['lookat'], *camera['up'], camera['fov']]).all()
            or not 0 < camera['fov'] < 180
            or len(camera['res']) != 2 or any(type(x) is not int or x < 1 for x in camera['res'])
            or np.linalg.norm(np.array(camera['pos'])-camera['lookat']) < .1
            or np.linalg.norm(np.cross(np.array(camera['lookat'])-camera['pos'],camera['up'])) < 1e-8):
        raise ValueError('Invalid overview camera')
    width,height = camera['res']
    positions = [project(m['world_xyz'],camera) for m in manifest['markers']]
    if any(not np.isfinite(p).all() or not 60 < p[0] < width-60 or not 100 < p[1] < height-70
           for p in positions):
        raise ValueError('Overview does not frame every landmark with readable label margins')
    if any(np.linalg.norm(np.array(a)-b)<60 for i,a in enumerate(positions) for b in positions[i+1:]):
        raise ValueError('Projected landmark labels cannot be distinguished')
