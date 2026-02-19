# ==============================================
# 导入必要的库
# ==============================================
import math          # 数学计算
import random        # 随机数生成
import sys           # 系统相关功能
from dataclasses import dataclass  # 数据类装饰器

import pygame        # 游戏开发库
from pygame.locals import DOUBLEBUF, OPENGL  # Pygame显示模式

import magiccube     # 魔方求解库
from magiccube import BasicSolver  # 基础求解器

from OpenGL.GL import *      # OpenGL基础函数
from OpenGL.GLU import gluPerspective, gluLookAt  # OpenGL实用函数


# ==============================
#  magiccube state: ULFRBD (len 54)
# ==============================
SOLVED_ULFRBD = "YYYYYYYYY" "RRRRRRRRR" "GGGGGGGGG" "OOOOOOOOO" "BBBBBBBBB" "WWWWWWWWW"
FACE_ORDER = "ULFRBD"

RGB = {
    "Y": (245, 230, 0),
    "W": (245, 245, 245),
    "R": (220, 30, 30),
    "O": (255, 140, 0),
    "G": (20, 170, 60),
    "B": (30, 90, 220),
}
PLASTIC = (25, 25, 30)

def ensure_state54(state) -> str:
    state = str(state).replace(" ", "").replace("\n", "").replace("\r", "")
    if len(state) != 54:
        raise ValueError(f"Cube state length={len(state)} expected 54.")
    return state


# ----------------- face coords mapping -----------------
def build_face_coords(size=3):
    r = list(range(size))
    rr = list(range(size - 1, -1, -1))
    return {
        "L": [[(0, y, z) for z in r] for y in rr],
        "R": [[(size - 1, y, z) for z in rr] for y in rr],
        "D": [[(x, 0, z) for x in r] for z in rr],
        "U": [[(x, size - 1, z) for x in r] for z in r],
        "B": [[(x, y, 0) for x in rr] for y in rr],
        "F": [[(x, y, size - 1) for x in r] for y in rr],
    }

FACE_COORDS = build_face_coords(3)


# ----------------- discrete 90° rotation -----------------
def rot_pos_90(pos, axis, sign):
    x, y, z = pos
    if axis == "x":
        return (x, -z, y) if sign == +1 else (x, z, -y)
    if axis == "y":
        return (z, y, -x) if sign == +1 else (-z, y, x)
    return (-y, x, z) if sign == +1 else (y, -x, z)

MAP_X_POS = {"U": "F", "F": "D", "D": "B", "B": "U", "L": "L", "R": "R"}
MAP_Y_POS = {"F": "R", "R": "B", "B": "L", "L": "F", "U": "U", "D": "D"}
MAP_Z_POS = {"U": "L", "L": "D", "D": "R", "R": "U", "F": "F", "B": "B"}

def invert_map(m):
    return {v: k for k, v in m.items()}

MAP_X_NEG = invert_map(MAP_X_POS)
MAP_Y_NEG = invert_map(MAP_Y_POS)
MAP_Z_NEG = invert_map(MAP_Z_POS)

def rot_face_label(face, axis, sign):
    if axis == "x":
        return (MAP_X_POS if sign == +1 else MAP_X_NEG)[face]
    if axis == "y":
        return (MAP_Y_POS if sign == +1 else MAP_Y_NEG)[face]
    return (MAP_Z_POS if sign == +1 else MAP_Z_NEG)[face]

FACE_AXIS_LAYER = {
    "U": ("y", +1),
    "D": ("y", -1),
    "R": ("x", +1),
    "L": ("x", -1),
    "F": ("z", +1),
    "B": ("z", -1),
}


# ==============================
#  Cube model (cubies + stickers)
# ==============================
@dataclass
class Cubie:
    pos: tuple
    stickers: dict

class CubeModel:
    def __init__(self, state_ulfrbd: str):
        self.cubies = {}
        self.from_state_ulfrbd(state_ulfrbd)

    def copy(self):
        c = CubeModel.__new__(CubeModel)
        c.cubies = {p: Cubie(p, dict(cb.stickers)) for p, cb in self.cubies.items()}
        return c

    def from_state_ulfrbd(self, state_ulfrbd: str):
        s = ensure_state54(state_ulfrbd)
        cubies = {}
        idx = 0
        for face in FACE_ORDER:
            grid = FACE_COORDS[face]
            for row in grid:
                for (x0, y0, z0) in row:
                    col = s[idx]; idx += 1
                    pos = (x0 - 1, y0 - 1, z0 - 1)
                    cubies.setdefault(pos, Cubie(pos, {})).stickers[face] = col
        self.cubies = cubies

    def to_state_ulfrbd(self) -> str:
        out = []
        for face in FACE_ORDER:
            grid = FACE_COORDS[face]
            for row in grid:
                for (x0, y0, z0) in row:
                    pos = (x0 - 1, y0 - 1, z0 - 1)
                    out.append(self.cubies[pos].stickers[face])
        return ensure_state54("".join(out))

    def apply_layer_quarter(self, axis, layer_val, sign):
        idx = "xyz".index(axis)
        affected = [cb for cb in self.cubies.values() if cb.pos[idx] == layer_val]
        moved = {}
        for cb in affected:
            new_pos = rot_pos_90(cb.pos, axis, sign)
            new_st = {rot_face_label(f, axis, sign): c for f, c in cb.stickers.items()}
            moved[new_pos] = Cubie(new_pos, new_st)
        for cb in affected:
            del self.cubies[cb.pos]
        for p, cb in moved.items():
            self.cubies[p] = cb


# ==============================
#  moves
# ==============================
def norm_token(t: str) -> str:
    t = t.strip()
    if t.endswith("i"):
        t = t[:-1] + "'"
    return t

def parse_token(t: str):
    t = norm_token(t)
    if not t:
        return "", False, 0
    base = t[0].upper()
    prime = ("'" in t)
    count = 2 if ("2" in t) else 1
    return base, prime, count

def random_scramble(n=25):
    faces = list("UDLRFB")
    out = []
    last = None
    for _ in range(n):
        f = random.choice([x for x in faces if x != last])
        last = f
        out.append(f + random.choice(["", "'", "2"]))
    return out


# ----------------- calibrate sign vs magiccube.rotate(face) -----------------
def calibrate_face_signs():
    base_state = SOLVED_ULFRBD
    base_model = CubeModel(base_state)
    face_sign = {}

    for face in "UDLRFB":
        axis, layer = FACE_AXIS_LAYER[face]
        ref = magiccube.Cube(3, base_state)
        ref.rotate(face)
        target = ensure_state54(ref.get())

        found = None
        for sign in (+1, -1):
            test = base_model.copy()
            test.apply_layer_quarter(axis, layer, sign)
            if test.to_state_ulfrbd() == target:
                found = sign
                break
        if found is None:
            raise RuntimeError(f"校准失败：{face} 无法匹配 magiccube.rotate({face})")
        face_sign[face] = found

    return face_sign


# ----------------- strip X/Y/Z to keep cube fixed (try 8 combos) -----------------
def apply_rot_to_face_map(face_map, rot_old2new):
    inv = invert_map(rot_old2new)
    return {f: face_map[inv[f]] for f in "UDLRFB"}

def convert_moves_fixed_orientation(raw_moves, face_sign, model: CubeModel):
    rot_choices = {
        "X": [MAP_X_POS, MAP_X_NEG],
        "Y": [MAP_Y_POS, MAP_Y_NEG],
        "Z": [MAP_Z_POS, MAP_Z_NEG],
    }

    def try_combo(rotX, rotY, rotZ):
        face_map = {f: f for f in "UDLRFB"}
        out = []
        for mv in raw_moves:
            s = norm_token(mv if isinstance(mv, str) else str(mv))
            if not s:
                continue
            base, prime, count = parse_token(s)

            if base in "XYZ":
                rot = {"X": rotX, "Y": rotY, "Z": rotZ}[base]
                if prime:
                    rot = invert_map(rot)
                for _ in range(count):
                    face_map = apply_rot_to_face_map(face_map, rot)
                continue

            if base in "UDLRFB":
                phys = face_map[base]
                for _ in range(count):
                    out.append(phys + ("'" if prime else ""))
        return out

    def apply_face_token(m: CubeModel, tok: str):
        b, p, cnt = parse_token(tok)
        if b not in "UDLRFB":
            return
        axis, layer = FACE_AXIS_LAYER[b]
        sg = face_sign[b] * (-1 if p else +1)
        for _ in range(cnt):
            m.apply_layer_quarter(axis, layer, +1 if sg > 0 else -1)

    best = None
    for rotX in rot_choices["X"]:
        for rotY in rot_choices["Y"]:
            for rotZ in rot_choices["Z"]:
                fixed = try_combo(rotX, rotY, rotZ)
                test = model.copy()
                for t in fixed:
                    apply_face_token(test, t)
                if test.to_state_ulfrbd() == SOLVED_ULFRBD:
                    return fixed
                best = fixed
    print("警告：无法找到能通过自检的 X/Y/Z 映射组合（可能出现宽转等）。")
    return best or []


# ==============================
#  animation
# ==============================
def clamp(x, a, b):
    return a if x < a else b if x > b else x

@dataclass
class TurnAnim:
    face: str
    prime: bool
    axis: str
    layer: int
    sign: int
    duration: float
    t: float = 0.0

    def angle_deg(self):
        u = clamp(self.t / self.duration, 0.0, 1.0)
        u = 0.5 - 0.5 * math.cos(math.pi * u)
        return 90.0 * u * self.sign

    def done(self):
        return self.t >= self.duration


# ==============================
#  OpenGL helpers
# ==============================
FACE_ID = {"U": 1, "D": 2, "L": 3, "R": 4, "F": 5, "B": 6}  # for picking

def setup_gl(w, h):
    glViewport(0, 0, w, h)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, w / float(h), 0.1, 200.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearDepth(1.0)

    # ✅ 稳：先禁用剔除，避免“顶点绕向不一致导致缺面”
    glDisable(GL_CULL_FACE)

    glShadeModel(GL_SMOOTH)
    glEnable(GL_NORMALIZE)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (6.0, 8.0, 10.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.20, 0.20, 0.20, 1.0))

    glDisable(GL_DITHER)

    try:
        glEnable(GL_MULTISAMPLE)
    except Exception:
        pass

def orbit_eye(yaw, pitch, radius):
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    x = radius * cp * sy
    y = radius * sp
    z = radius * cp * cy
    return (x, y, z)

def face_axis_vec(axis: str):
    return (1.0, 0.0, 0.0) if axis == "x" else (0.0, 1.0, 0.0) if axis == "y" else (0.0, 0.0, 1.0)

def set_material_rgb255(rgb, ambient_scale=0.30):
    r, g, b = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  (r, g, b, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,  (r*ambient_scale, g*ambient_scale, b*ambient_scale, 1.0))
    # ✅ 更亮的高光
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.85, 0.85, 0.85, 1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 80.0)
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))

def draw_cubelet(stickers: dict,
                 size=0.92,
                 sticker_inset=0.18,
                 sticker_lift=0.03,
                 pick_mode=False):
    """
    Draw a single cubelet centered at origin.
    - Normal pass: plastic + stickers with lighting
    - Pick pass: stickers only, encoded face id colors (no lighting)
    """
    half = size * 0.5

    def quad(n, v0, v1, v2, v3):
        glNormal3f(*n)
        glBegin(GL_QUADS)
        glVertex3f(*v0); glVertex3f(*v1); glVertex3f(*v2); glVertex3f(*v3)
        glEnd()

    faces = {
        "U": ((0, +1, 0), (-half, +half, -half), (+half, +half, -half), (+half, +half, +half), (-half, +half, +half)),
        "D": ((0, -1, 0), (-half, -half, +half), (+half, -half, +half), (+half, -half, -half), (-half, -half, -half)),
        "L": ((-1, 0, 0), (-half, -half, +half), (-half, -half, -half), (-half, +half, -half), (-half, +half, +half)),
        "R": ((+1, 0, 0), (+half, -half, -half), (+half, -half, +half), (+half, +half, +half), (+half, +half, -half)),
        "F": ((0, 0, +1), (-half, -half, +half), (+half, -half, +half), (+half, +half, +half), (-half, +half, +half)),
        "B": ((0, 0, -1), (+half, -half, -half), (-half, -half, -half), (-half, +half, -half), (+half, +half, -half)),
    }

    # plastic
    if not pick_mode:
        set_material_rgb255(PLASTIC, ambient_scale=0.35)
        for _, (n, a, b, c, d) in faces.items():
            quad(n, a, b, c, d)

    # stickers
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-1.0, -1.0)

    for f, col_char in stickers.items():
        n, a, b, c, d = faces[f]

        def inset_vertex(v):
            x, y, z = v
            nx, ny, nz = n
            sx = 1.0 - (sticker_inset if nx == 0 else 0.0)
            sy = 1.0 - (sticker_inset if ny == 0 else 0.0)
            sz = 1.0 - (sticker_inset if nz == 0 else 0.0)
            x, y, z = x * sx, y * sy, z * sz
            x += nx * sticker_lift
            y += ny * sticker_lift
            z += nz * sticker_lift
            return (x, y, z)

        aa, bb, cc, dd = map(inset_vertex, (a, b, c, d))

        if pick_mode:
            fid = FACE_ID.get(f, 0)
            glColor3ub(fid * 30, 0, 0)
            quad(n, aa, bb, cc, dd)
        else:
            set_material_rgb255(RGB.get(col_char, (200, 0, 200)), ambient_scale=0.25)
            quad(n, aa, bb, cc, dd)

    glDisable(GL_POLYGON_OFFSET_FILL)


def render_scene(model: CubeModel, yaw, pitch, cam_radius, anim: TurnAnim | None, pick_mode=False):
    """渲染整个魔方场景"""
    # 设置背景颜色（深灰色）
    glClearColor(22/255.0, 22/255.0, 28/255.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 强制恢复状态：避免拾取过程影响正常渲染
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)

    # 根据模式设置光照
    if pick_mode:
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
    else:
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

    # 设置摄像机视角
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    ex, ey, ez = orbit_eye(yaw, pitch, cam_radius)
    gluLookAt(ex, ey, ez, 0, 0, 0, 0, 1, 0)  # 摄像机看向原点

    # 整体缩放魔方
    glPushMatrix()
    glScalef(1.35, 1.35, 1.35)

    # 绘制每个立方块
    for cb in model.cubies.values():
        glPushMatrix()

        # 应用动画旋转（如果当前立方块在旋转层中）
        if anim and cb.pos["xyz".index(anim.axis)] == anim.layer:
            ax = face_axis_vec(anim.axis)
            glRotatef(anim.angle_deg(), ax[0], ax[1], ax[2])

        # 平移到立方块位置
        x, y, z = cb.pos
        glTranslatef(float(x), float(y), float(z))
        
        # 绘制立方块
        draw_cubelet(cb.stickers, pick_mode=pick_mode)
        glPopMatrix()

    glPopMatrix()
    glFlush()  # 强制执行OpenGL命令


def pick_face_under_mouse(model, yaw, pitch, cam_radius, anim, mouse_xy, w, h):
    """
    颜色拾取：用贴纸颜色编码 face id
    注意：拾取时关闭多重采样/抖动，避免颜色被“边缘混合”污染
    """
    msaa_was = glIsEnabled(GL_MULTISAMPLE)
    dither_was = glIsEnabled(GL_DITHER)

    if msaa_was:
        glDisable(GL_MULTISAMPLE)
    if dither_was:
        glDisable(GL_DITHER)

    glReadBuffer(GL_BACK)
    render_scene(model, yaw, pitch, cam_radius, anim, pick_mode=True)
    glFlush()

    mx, my = mouse_xy
    ry = h - my - 1
    px = glReadPixels(mx, ry, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)

    # restore
    if msaa_was:
        glEnable(GL_MULTISAMPLE)
    if dither_was:
        glEnable(GL_DITHER)

    if px is None:
        return None

    # PyOpenGL 在不同平台可能返回 bytes 或 ndarray-like
    if isinstance(px, (bytes, bytearray)):
        r = px[0]
    else:
        r = int(px[0][0][0])

    if r == 0:
        return None

    fid = int(round(r / 30.0))
    for face, _id in FACE_ID.items():
        if _id == fid:
            return face
    return None


# ==============================
#  main
# ==============================
def main():
    pygame.init()
    w, h = 1180, 780

    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D 魔方（pygame+OpenGL）- 专业鼠标手势版")

    clock = pygame.time.Clock()
    setup_gl(w, h)

    # logic init
    tmp = CubeModel(SOLVED_ULFRBD)
    if tmp.to_state_ulfrbd() != SOLVED_ULFRBD:
        raise SystemExit("ULFRBD<->模型转换不一致（程序错误）")

    face_sign = calibrate_face_signs()
    model = CubeModel(SOLVED_ULFRBD)

    # camera
    yaw, pitch = -0.7, 0.55
    cam_radius = 10.0

    # anim queue
    queue = []
    anim = None
    paused = False
    speed = 0.22
    last_move = ""

    def enqueue(tok: str):
        nonlocal queue
        tok = norm_token(tok)
        base, prime, count = parse_token(tok)
        if base not in "UDLRFB":
            return
        for _ in range(count):
            queue.append(base + ("'" if prime else ""))

    def start_next():
        nonlocal anim, last_move
        if anim or not queue:
            return
        tok = queue.pop(0)
        base, prime, _ = parse_token(tok)
        axis, layer = FACE_AXIS_LAYER[base]
        sg = face_sign[base] * (-1 if prime else +1)
        anim = TurnAnim(base, prime, axis, layer, +1 if sg > 0 else -1, speed)
        last_move = tok

    def commit():
        nonlocal anim
        if not anim:
            return
        model.apply_layer_quarter(anim.axis, anim.layer, anim.sign)
        anim = None

    def do_reset():
        nonlocal model, queue, anim, last_move
        model = CubeModel(SOLVED_ULFRBD)
        queue.clear()
        anim = None
        last_move = ""

    def do_solve():
        nonlocal queue, anim, paused
        paused = False
        queue.clear()
        anim = None

        state = model.to_state_ulfrbd()
        try:
            mc = magiccube.Cube(3, state, hist=True)
        except TypeError:
            mc = magiccube.Cube(3, state)

        solver = BasicSolver(mc)
        solver.solve()

        try:
            hist = mc.history(to_str=True)
        except TypeError:
            hist = mc.history()

        raw_moves = hist.split() if isinstance(hist, str) else [str(x) for x in hist]
        fixed = convert_moves_fixed_orientation(raw_moves, face_sign, model)
        for t in fixed:
            enqueue(t)

    def apply_face_turn(face, direction, ctrl=False):
        if face not in "UDLRFB":
            return
        if ctrl:
            enqueue(face + "2")
        else:
            enqueue(face if direction > 0 else (face + "'"))

    # ==============================
    #  专业鼠标手势状态机
    # ==============================
    # 点击判定阈值：短按 + 小位移
    CLICK_MAX_MS = 260
    CLICK_MOVE_PX = 6

    # 双键 Orbit：一旦进入，直到两键都松开前，完全屏蔽“点击拧面”
    suppress_until_all_up = False
    gesture = "idle"   # idle / clickL / clickR / orbit

    left_down = False
    right_down = False

    down_pos = (0, 0)
    down_time = 0
    click_dragged = False
    last_mouse = (0, 0)

    # 视角灵敏度（你觉得反向就把符号改掉）
    ORBIT_SENS = 0.006

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                shift = bool(mods & pygame.KMOD_SHIFT)
                ctrl = bool(mods & pygame.KMOD_CTRL)

                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_BACKSPACE, pygame.K_x):
                    do_reset()
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = max(0.05, speed - 0.03)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = min(1.0, speed + 0.03)
                elif e.key == pygame.K_s:
                    queue.clear()
                    anim = None
                    paused = False
                    for m in random_scramble(25):
                        enqueue(m)
                elif e.key == pygame.K_RETURN:
                    do_solve()
                else:
                    keymap = {
                        pygame.K_u: "U",
                        pygame.K_d: "D",
                        pygame.K_l: "L",
                        pygame.K_r: "R",
                        pygame.K_f: "F",
                        pygame.K_b: "B",
                    }
                    if e.key in keymap:
                        base = keymap[e.key]
                        tok = base + ("2" if ctrl else ("'" if shift else ""))
                        paused = False
                        enqueue(tok)

            # 滚轮拧面：悬停在面上
            elif e.type == pygame.MOUSEWHEEL:
                mods = pygame.key.get_mods()
                ctrl = bool(mods & pygame.KMOD_CTRL)

                # 如果正在双键 orbit（或刚进入），仍然允许滚轮：你想禁掉也可以
                face = pick_face_under_mouse(model, yaw, pitch, cam_radius, anim, mouse, w, h)
                if face:
                    step = 1 if e.y > 0 else -1
                    for _ in range(abs(int(e.y)) or 1):
                        apply_face_turn(face, step, ctrl=ctrl)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    left_down = True
                elif e.button == 3:
                    right_down = True
                else:
                    continue

                down_pos = mouse
                down_time = pygame.time.get_ticks()
                click_dragged = False
                last_mouse = mouse

                # 进入双键：立即切换 orbit，并屏蔽点击直到两键都松开
                if left_down and right_down:
                    gesture = "orbit"
                    suppress_until_all_up = True
                else:
                    gesture = "clickL" if left_down else ("clickR" if right_down else "idle")

            elif e.type == pygame.MOUSEMOTION:
                mx, my = mouse
                lx, ly = last_mouse
                dx, dy = mx - lx, my - ly

                # 位移判定
                if abs(mx - down_pos[0]) + abs(my - down_pos[1]) > CLICK_MOVE_PX:
                    click_dragged = True

                # 双键 orbit：只有两键仍按下才旋转
                if gesture == "orbit" and left_down and right_down:
                    # ✅ 你说“反向”：专业默认是“鼠标往右拖 => 视角往右转”
                    yaw -= dx * ORBIT_SENS
                    pitch += dy * ORBIT_SENS
                    pitch = clamp(pitch, -1.25, 1.25)

                last_mouse = mouse

            elif e.type == pygame.MOUSEBUTTONUP:
                # 先更新键状态
                if e.button == 1:
                    left_down = False
                elif e.button == 3:
                    right_down = False
                else:
                    continue

                # ✅ 专业规则：只要进入过双键 orbit，就一直屏蔽“点击拧面”
                if suppress_until_all_up:
                    if (not left_down) and (not right_down):
                        suppress_until_all_up = False
                        gesture = "idle"
                    # 直接跳过点击拧面
                    continue

                # 单键点击拧面：必须满足短按+小位移，并且释放的是对应按钮
                t_up = pygame.time.get_ticks()
                is_click = (not click_dragged) and ((t_up - down_time) <= CLICK_MAX_MS)

                mods = pygame.key.get_mods()
                ctrl = bool(mods & pygame.KMOD_CTRL)

                if is_click:
                    if e.button == 1:
                        face = pick_face_under_mouse(model, yaw, pitch, cam_radius, anim, mouse, w, h)
                        if face:
                            apply_face_turn(face, +1, ctrl=ctrl)
                    elif e.button == 3:
                        face = pick_face_under_mouse(model, yaw, pitch, cam_radius, anim, mouse, w, h)
                        if face:
                            apply_face_turn(face, -1, ctrl=ctrl)

                gesture = "idle"

        # animate
        if not paused:
            if anim is None and queue:
                start_next()
            if anim is not None:
                anim.duration = speed
                anim.t += dt
                if anim.done():
                    commit()

        # render visible pass
        render_scene(model, yaw, pitch, cam_radius, anim, pick_mode=False)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
