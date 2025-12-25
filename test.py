import time
import math
from pathlib import Path
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import cv2
import mediapipe as mp
from collections import deque


# =========================
# НАСТРОЙКИ / КОНСТАНТЫ
# =========================

MODEL_PATH = Path("./hand_landmarker.task")  # положи рядом со скриптом

d_traj = deque(maxlen=120)
yo_traj = deque(maxlen=120)
z_traj = deque(maxlen=220)

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
FONT = ImageFont.truetype(FONT_PATH, 42)

# Связи 21 точек (как в classic MediaPipe Hands) для отрисовки "скелета"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17)              # palm links
]

# --- Пороги (их потом подгоняешь под себя/камеру) ---
# Чем меньше пороги углов — тем "сильнее согнут" палец должен быть.
PIP_DIP_CURLED_THR = 165    # если < этого угла -> палец считаем согнутым
THUMB_IP_FLEX_THR = 165     # большой согнут, если угол в IP < этого

# Нормированные расстояния (делим на "размер ладони", чтобы не зависеть от масштаба)
TIP_TO_MCP_CURLED_THR = 0.75   # tip рядом со своей "основанием" (MCP) -> палец согнут
FIST_COMPACT_THR = 1.10        # все пальцы "в куче" (tips близко к запястью) — доп.стабилизация

THUMB_TUCK_TOUCH_THR = 0.90    # кончик большого рядом с кулаком (например, около index/middle MCP)

# Сглаживание результата: храним последние N кадров, берём большинство
HIST_LEN = 7
HIST_MIN_VOTES = 4

DEBUG = False  # True -> показывать числа (углы/дистанции)

# =========================
# УТИЛИТА
# =========================

def put_text_ru(frame_bgr, text, org, bgr=(0, 255, 0)):
    """
    Рисует Unicode-текст (кириллица) поверх кадра.
    org = (x, y) левый верх.
    bgr = цвет как в OpenCV.
    """
    # BGR -> RGB -> PIL
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # PIL ждёт RGB цвет
    rgb = (bgr[2], bgr[1], bgr[0])
    draw.text(org, text, font=FONT, fill=rgb)

    # PIL -> обратно в OpenCV BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# =========================
# УТИЛИТЫ МАТЕМАТИКИ
# =========================

def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cosv = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosv))

def angle_3d(a, b, c):
    """Угол ABC (в градусах) по 3 точкам a,b,c. a,b,c = (x,y,z)."""
    bax = a[0] - b[0]; bay = a[1] - b[1]; baz = a[2] - b[2]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]; bcz = c[2] - b[2]
    dot = bax*bcx + bay*bcy + baz*bcz
    na = math.sqrt(bax*bax + bay*bay + baz*baz) + 1e-9
    nc = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz) + 1e-9
    cosv = max(-1.0, min(1.0, dot/(na*nc)))
    return math.degrees(math.acos(cosv))

def dist_3d(a, b):
    """Евклидово расстояние между (x,y,z)."""
    dx = a[0]-b[0]; dy = a[1]-b[1]; dz = a[2]-b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def lms_to_xyz(hand_landmarks):
    """hand_landmarks: список из 21 landmark -> [(x,y,z), ...]"""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

def palm_scale(xyz):
    """
    Масштаб ладони (для нормализации расстояний).
    Берём расстояние wrist(0) -> middle_mcp(9) (стабильная базовая длина).
    """
    return dist_3d(xyz[0], xyz[9]) + 1e-9

def ndist(xyz, i, j):
    """Нормированное расстояние между точками i и j: dist / scale."""
    s = palm_scale(xyz)
    return dist_3d(xyz[i], xyz[j]) / s

def lm_to_xyz(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

def dist2(p, q):
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def hand_scale(xyz):
    return dist2(xyz[0], xyz[9]) + 1e-9

def finger_extended(xyz, mcp, pip, dip, tip, thr=170):
    ang_pip = angle_3d(xyz[mcp], xyz[pip], xyz[dip])
    ang_dip = angle_3d(xyz[pip], xyz[dip], xyz[tip])
    return (ang_pip > thr) and (ang_dip > thr)

def palm_is_sideways(xyz):
    """
    True, если ладонь повернута БОКОМ к камере,
    False — если ладонь смотрит в камеру.
    """
    p0 = np.array(xyz[0])   # wrist
    p5 = np.array(xyz[5])   # index MCP
    p17 = np.array(xyz[17]) # pinky MCP

    # два вектора в плоскости ладони
    v1 = p5 - p0
    v2 = p17 - p0

    # нормаль к плоскости ладони
    normal = np.cross(v1, v2)

    # нас интересует соотношение X и Z
    nx, ny, nz = abs(normal[0]), abs(normal[1]), abs(normal[2])

    # если Z доминирует → ладонь к камере ❌
    # если X доминирует → ладонь боком ✅
    return nx > nz

def vec_0_17_along_y(xyz, max_angle_deg=12, enforce_up=True, min_len_nd=0.30):
    """
    True, если вектор wrist(0)->pinky_mcp(17) почти вертикален (вдоль оси Y).

    max_angle_deg: чем меньше — тем строже вертикаль
    enforce_up=True: требует, чтобы 17 был ВЫШЕ 0 (dy < 0), т.е. "вверх"
    min_len_nd: защита от шума, если рука слишком далеко/плохо детектится
    """
    dx = xyz[17][0] - xyz[0][0]
    dy = xyz[17][1] - xyz[0][1]

    if enforce_up and not (dy < 0):
        return False

    # нормированная длина (чтобы не ловить мусор при плохом детекте)
    s = palm_scale(xyz)
    vlen = math.sqrt(dx*dx + dy*dy) / (s + 1e-9)
    if vlen < min_len_nd:
        return False

    # угол к оси Y: cos = |dy| / |v|
    cosv = abs(dy) / (math.sqrt(dx*dx + dy*dy) + 1e-9)
    cosv = max(-1.0, min(1.0, cosv))
    ang = math.degrees(math.acos(cosv))
    return ang <= max_angle_deg


def wrist_not_bent(xyz, thr_z=0.07, thr_ny=0.60):
    """
    Приближённая проверка "кисть не согнута":
    - ладонь не наклонена сильно вверх/вниз (|normal_y| небольшой)
    - запястье по Z близко к костяшкам (нет сильного залома к/от камеры)

    thr_z  подстройка 0.05–0.10
    thr_ny подстройка 0.45–0.75 (меньше = строже)
    """
    p0 = np.array(xyz[0])   # wrist
    p5 = np.array(xyz[5])   # index MCP
    p17 = np.array(xyz[17]) # pinky MCP

    v1 = p5 - p0
    v2 = p17 - p0
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-9)

    ok_pitch = abs(normal[1]) < thr_ny

    kn_z = (xyz[5][2] + xyz[9][2] + xyz[13][2] + xyz[17][2]) / 4.0
    ok_z = abs(xyz[0][2] - kn_z) < thr_z

    return ok_pitch and ok_z
# =========================
# ОТРИСОВКА
# =========================

def draw_hand(frame_bgr, hand_landmarks_norm, color=(0, 255, 0)):
    """Рисуем точки + связи."""
    h, w = frame_bgr.shape[:2]
    pts = []
    for lm in hand_landmarks_norm:
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))
        cv2.circle(frame_bgr, (x, y), 4, color, -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], color, 2)


# =========================
# ПРИЗНАКИ ПАЛЬЦЕВ
# =========================

def finger_curled(xyz, mcp, pip, dip, tip):
    """
    Палец считаем согнутым, если:
    1) углы в PIP и DIP меньше порога (палец не "прямой"),
    2) tip близко к MCP (кулак/сжатие).
    """
    ang_pip = angle_3d(xyz[mcp], xyz[pip], xyz[dip])
    ang_dip = angle_3d(xyz[pip], xyz[dip], xyz[tip])
    close_tip = ndist(xyz, tip, mcp) < TIP_TO_MCP_CURLED_THR

    return (ang_pip < PIP_DIP_CURLED_THR) and (ang_dip < PIP_DIP_CURLED_THR) and close_tip, ang_pip, ang_dip, ndist(xyz, tip, mcp)

def thumb_flexed_and_tucked(xyz):
    """
    Для твоего случая "А": большой палец Согнут, tip(4) близко к кулаку.
    Учитываем, что точки 3 и 4 могут быть очень близко (tip согнут).
    """
    # угол в IP сустава большого: 2-3-4
    ang_ip = angle_3d(xyz[2], xyz[3], xyz[4])

    # tip большого должен быть "рядом с кулаком" — около index_mcp(5) или middle_mcp(9)
    d_to_index_mcp = ndist(xyz, 4, 5)
    d_to_middle_mcp = ndist(xyz, 4, 9)
    tucked = min(d_to_index_mcp, d_to_middle_mcp) < THUMB_TUCK_TOUCH_THR

    flexed = ang_ip < THUMB_IP_FLEX_THR
    return (flexed and tucked), ang_ip, min(d_to_index_mcp, d_to_middle_mcp)


# =========================
# ЖЕСТЫ (ПРАВИЛА)
# =========================

def is_letter_A(hand_landmarks):
    """
    Буква "А" (по твоему описанию):
    - четыре пальца (index/middle/ring/pinky) согнуты в кулак
    - большой палец согнут и "прижат" к кулаку (tip близко к MCP указательного/среднего)
    """
    xyz = lms_to_xyz(hand_landmarks)

    idx_ok, idx_pip, idx_dip, idx_tipmcp = finger_curled(xyz, 5, 6, 7, 8)
    mid_ok, mid_pip, mid_dip, mid_tipmcp = finger_curled(xyz, 9, 10, 11, 12)
    ring_ok, ring_pip, ring_dip, ring_tipmcp = finger_curled(xyz, 13, 14, 15, 16)
    pink_ok, pink_pip, pink_dip, pink_tipmcp = finger_curled(xyz, 17, 18, 19, 20)

    thumb_ok, thumb_ip, thumb_touch = thumb_flexed_and_tucked(xyz)

    if not thumb_ok:
        return False

    # Доп. стабилизация: tips (8,12,16,20) не должны быть "слишком далеко" от запястья (0)
    fist_compact = (
        ndist(xyz, 8, 0) < FIST_COMPACT_THR and
        ndist(xyz, 12, 0) < FIST_COMPACT_THR and
        ndist(xyz, 16, 0) < FIST_COMPACT_THR and
        ndist(xyz, 20, 0) < FIST_COMPACT_THR
    )

    if not (idx_ok and mid_ok and ring_ok and pink_ok):
        return False

    ok = idx_ok and mid_ok and ring_ok and pink_ok and thumb_ok and fist_compact

    if not (
        ndist(xyz, 8, 5)   < 0.75 and  # NEW
        ndist(xyz, 12, 9)  < 0.75 and  # NEW
        ndist(xyz, 16, 13) < 0.75 and  # NEW
        ndist(xyz, 20, 17) < 0.75      # NEW
    ):
        return False
    
    knuckles_y = (xyz[5][1] + xyz[9][1] + xyz[13][1] + xyz[17][1]) / 4.0
    hand_up = (knuckles_y + 0.01) < xyz[0][1]   # NEW: +0.02 — допуск/запас от дрожания
    if not hand_up:
        return False
    
    return ok


def is_letter_B(hand_landmarks):
    """
    Буква "Б" (по твоему описанию):
    - указательный прямой
    - средний согнут и прижат к указательному
    - безымянный и мизинец согнуты дугой
    - кончик большого касается (примерно) кончиков/области безымянного и мизинца (16 и 20)
    - кисть поднята вверх (примерно: tip указательного выше запястья)
    """
    xyz = lms_to_xyz(hand_landmarks)

    # 1) Указательный прямой
    index_ext = finger_extended(xyz, 5, 6, 7, 8, thr=170)

    # 2) Средний НЕ прямой + прижат к указательному
    middle_ext = finger_extended(xyz, 9, 10, 11, 12, thr=170)
    middle_not_ext = not middle_ext

    # "прижат": tip среднего (12) близко к суставам указательного (6 или 7)
    d12_6 = ndist(xyz, 12, 6)
    d12_7 = ndist(xyz, 12, 7)
    middle_touch_index = min(d12_6, d12_7) < 0.65  # подгонка

    # 3) Безымянный и мизинец согнуты
    ring_curled,  ring_pip,  ring_dip,  ring_tipmcp  = finger_curled(xyz, 13, 14, 15, 16)
    pinky_curled, pink_pip,  pink_dip,  pink_tipmcp  = finger_curled(xyz, 17, 18, 19, 20)

    # 4) Большой касается области 16 и 20 (кончики/дуга)
    d4_16 = ndist(xyz, 4, 16)
    d4_20 = ndist(xyz, 4, 20)
    thumb_touch = (d4_16 < 0.6) or (d4_20 < 0.65)  # подгонка: +-0.1

    # 5) "Кисть вверх" (очень мягко)
    hand_up = (xyz[8][1] + 0.02) < xyz[0][1]

    ok = (
        index_ext and
        middle_not_ext and middle_touch_index and
        ring_curled and pinky_curled and
        thumb_touch and
        hand_up
    )

    return ok

def is_letter_V(hand_landmarks):
    xyz = lms_to_xyz(hand_landmarks)

    # 1. Все 4 пальца прямые
    idx_ext   = finger_extended(xyz, 5, 6, 7, 8,  thr=170)
    mid_ext   = finger_extended(xyz, 9,10,11,12, thr=170)
    ring_ext  = finger_extended(xyz,13,14,15,16, thr=170)
    pink_ext  = finger_extended(xyz,17,18,19,20, thr=170)

    fingers_straight = idx_ext and mid_ext and ring_ext and pink_ext

    # 2. Большой палец не торчит (не "Г")
    # tip(4) не далеко от ладони (0)
    thumb_not_out = ndist(xyz, 4, 0) < 1.2

    # 3. Ладонь "боком": пальцы примерно в одной Z-плоскости
    z_vals = [xyz[i][2] for i in (8, 12, 16, 20)]
    z_spread = max(z_vals) - min(z_vals)
    palm_sideways = z_spread < 0.10   # подстройка: 0.08–0.15
    sideways = palm_is_sideways(xyz)

    return fingers_straight and thumb_not_out and palm_sideways and sideways

def is_letter_G(hand_landmarks):
    xyz = lms_to_xyz(hand_landmarks)

    # 1. Указательный прямой
    index_ext = finger_extended(xyz, 5, 6, 7, 8, thr=155)

    # 2. Большой палец прямой
    thumb_ext = finger_extended(xyz, 1, 2, 3, 4, thr=145)

    # 3. Остальные пальцы согнуты
    middle_ok = not finger_extended(xyz, 9, 10, 11, 12, thr=160)
    ring_ok   = not finger_extended(xyz,13,14,15,16, thr=160)
    pinky_ok  = not finger_extended(xyz,17,18,19,20, thr=160)

    # 4. Угол между большим и указательным (НАСТОЯЩИЙ)
    v_thumb = np.array(xyz[4]) - np.array(xyz[2])   # большой
    v_index = np.array(xyz[8]) - np.array(xyz[5])   # указательный

    angle_thumb_index = angle_between(v_thumb, v_index)
    right_angle = 60 < angle_thumb_index < 150

    # 5. Рука "указывает вниз"
    # направление указательного пальца
    v_index = np.array(xyz[8]) - np.array(xyz[5])

    # Y растёт вниз → значит палец направлен вниз
    hand_down = v_index[1] > 0.25 * abs(v_index[0])

    return (
        index_ext and
        thumb_ext and
        middle_ok and ring_ok and pinky_ok and
        right_angle and
        hand_down
    )

def is_D_pose(xyz):
    """
    Статическая форма буквы Д:
    - указательный и средний прямые
    - они рядом
    - остальные пальцы не прямые
    """
    idx_ext = finger_extended(xyz, 5, 6, 7, 8, thr=160)
    mid_ext = finger_extended(xyz, 9,10,11,12, thr=160)

    # указательный и средний вместе
    fingers_together = ndist(xyz, 8, 12) < 0.35

    ring_not_ext  = not finger_extended(xyz,13,14,15,16, thr=160)
    pinky_not_ext = not finger_extended(xyz,17,18,19,20, thr=160)

    return (
        idx_ext and
        mid_ext and
        fingers_together and
        ring_not_ext and
        pinky_not_ext
    )

def update_D_traj(d_traj, xyz, track_point=9):
    """
    Добавляет точку в траекторию Д.
    track_point = 9 (MCP среднего пальца) — стабильно.
    """
    x, y = xyz[track_point][0], xyz[track_point][1]
    d_traj.append((x, y))

def is_letter_D(d_traj, min_points=40):
    """
    Проверяет, что траектория описывает >= 2 оборотов.
    """
    if len(d_traj) < min_points:
        return False

    # центр траектории
    cx = sum(p[0] for p in d_traj) / len(d_traj)
    cy = sum(p[1] for p in d_traj) / len(d_traj)

    angles = []
    for x, y in d_traj:
        angles.append(math.atan2(y - cy, x - cx))

    total_angle = 0.0
    for i in range(1, len(angles)):
        da = angles[i] - angles[i - 1]
        if da > math.pi:
            da -= 2 * math.pi
        elif da < -math.pi:
            da += 2 * math.pi
        total_angle += da

    rotations = abs(total_angle) / (2 * math.pi)

    return rotations >= 1.8

def is_letter_E(hand_landmarks):
    """
    Буква "Е":
    - все пальцы согнуты
    - кончики пальцев образуют "туннель" (близко друг к другу)
    - ладонь поднята вверх
    """
    xyz = lms_to_xyz(hand_landmarks)

    # 1. Все пальцы (кроме большого) согнуты
    idx_ok, _, _, _   = finger_curled(xyz, 5, 6, 7, 8)
    mid_ok, _, _, _   = finger_curled(xyz, 9, 10, 11, 12)
    ring_ok, _, _, _  = finger_curled(xyz, 13, 14, 15, 16)
    pink_ok, _, _, _  = finger_curled(xyz, 17, 18, 19, 20)

    fingers_curled = idx_ok and mid_ok and ring_ok and pink_ok

    # 2. Большой палец согнут и "внутри"
    thumb_ok, _, _ = thumb_flexed_and_tucked(xyz)

    # 3. "Туннель": кончики пальцев близко друг к другу
    # Проверяем компактность дуги
    tunnel = (
        ndist(xyz, 8, 12) < 0.55 and
        ndist(xyz, 12, 16) < 0.55 and
        ndist(xyz, 16, 20) < 0.55
    )

    # 4. Рука поднята вверх:
    # средняя точка пальцев выше запястья
    avg_finger_y = (xyz[8][1] + xyz[12][1] + xyz[16][1] + xyz[20][1]) / 4
    hand_up = avg_finger_y + 0.02 < xyz[0][1]

    return fingers_curled and thumb_ok and tunnel and hand_up

def is_YO_pose(xyz):
    """
    Статическая форма для "Ё" = "туннель" ладонью:
    - 4 пальца согнуты
    - большой согнут/поджат
    - кончики пальцев близко друг к другу (компактная "дуга/туннель")
    """
    # 4 пальца согнуты
    idx_ok, _, _, _   = finger_curled(xyz, 5, 6, 7, 8)
    mid_ok, _, _, _   = finger_curled(xyz, 9, 10, 11, 12)
    ring_ok, _, _, _  = finger_curled(xyz, 13, 14, 15, 16)
    pink_ok, _, _, _  = finger_curled(xyz, 17, 18, 19, 20)
    if not (idx_ok and mid_ok and ring_ok and pink_ok):
        return False

    # большой палец поджат
    thumb_ok, _, _ = thumb_flexed_and_tucked(xyz)
    if not thumb_ok:
        return False

    # "туннель": кончики близко друг к другу
    tunnel = (
        ndist(xyz, 8, 12)  < 0.60 and
        ndist(xyz, 12, 16) < 0.60 and
        ndist(xyz, 16, 20) < 0.60 and
        ndist(xyz, 8, 20)  < 0.95
    )

    return tunnel


def update_YO_traj(yo_traj, xyz):
    """
    Сохраняем угол "поворота кисти" в кадре.
    Берём линию поперёк ладони: index_mcp(5) -> pinky_mcp(17).
    При вращении кисти (roll) эта линия заметно вращается.
    """
    x1, y1 = xyz[5][0], xyz[5][1]
    x2, y2 = xyz[17][0], xyz[17][1]
    ang = math.atan2((y2 - y1), (x2 - x1))  # [-pi..pi]
    yo_traj.append(ang)


def is_letter_YO(yo_traj, min_points=28, min_total_turn_rad=math.pi * 1.05):
    """
    Динамика для "Ё": пока держим "туннель", суммарная "закрутка" угла
    должна превысить порог (по умолчанию ~189°).

    min_total_turn_rad:
      - 0.8*pi  (~144°)  легче распознаётся, но больше ложных
      - 1.0*pi  (~180°)  норм
      - 1.2*pi  (~216°)  строже
    """
    if len(yo_traj) < min_points:
        return False

    total = 0.0
    for i in range(1, len(yo_traj)):
        da = yo_traj[i] - yo_traj[i - 1]
        # unwrap
        if da > math.pi:
            da -= 2 * math.pi
        elif da < -math.pi:
            da += 2 * math.pi
        total += abs(da)

    return total >= min_total_turn_rad

def is_letter_ZH(hand_landmarks):
    """
    Буква "Ж" (как на фото): "пучок/клюв"
    - 4 пальца вытянуты (или почти)
    - кончики 8,12,16,20 сильно сведены (пучок)
    - большой 4 подтянут к этому пучку (треугольник)
    - кисть НЕ согнута (wrist_not_bent)
    - (0 -> 17) почти строго вдоль оси Y (вертикально вверх)
    """
    xyz = lms_to_xyz(hand_landmarks)

    # NEW: вектор (0->17) строго по Y (вертикаль), и "вверх"
    if not vec_0_17_along_y(xyz, max_angle_deg=12, enforce_up=True, min_len_nd=0.30):
        return False

    # NEW: кисть не заломана в запястье
    if not wrist_not_bent(xyz, thr_z=0.07, thr_ny=0.60):
        return False

    # 1) 4 пальца вытянуты (допускаем небольшой изгиб)
    idx_ext  = finger_extended(xyz, 5, 6, 7, 8,  thr=155)
    mid_ext  = finger_extended(xyz, 9,10,11,12, thr=155)
    ring_ext = finger_extended(xyz,13,14,15,16, thr=155)
    pink_ext = finger_extended(xyz,17,18,19,20, thr=155)
    if not (idx_ext and mid_ext and ring_ext and pink_ext):
        return False

    # 2) Пучок кончиков (как на фото — очень плотный)
    tips_together = (
        ndist(xyz, 8, 12)  < 0.34 and
        ndist(xyz, 12, 16) < 0.34 and
        ndist(xyz, 16, 20) < 0.34 and
        ndist(xyz, 8, 20)  < 0.55
    )
    if not tips_together:
        return False

    # 3) Большой подтянут к пучку (создаёт "клюв/треугольник")
    thumb_near_cluster = min(
        ndist(xyz, 4, 8),
        ndist(xyz, 4, 12),
        ndist(xyz, 4, 16),
        ndist(xyz, 4, 20),
    ) < 0.55
    if not thumb_near_cluster:
        return False

    # 4) Стабилизация: пучок реально "сжат" относительно ширины ладони
    tip_spread = (ndist(xyz, 8, 12) + ndist(xyz, 12, 16) + ndist(xyz, 16, 20)) / 3.0
    mcp_spread = (ndist(xyz, 5, 9) + ndist(xyz, 9, 13) + ndist(xyz, 13, 17)) / 3.0
    if not (tip_spread < 0.65 * mcp_spread):
        return False

    # 5) (опционально, но на фото обычно так) ладонь чуть боком
    # если начнёт мешать при фронтальной постановке — убери эту строку
    if not palm_is_sideways(xyz):
        return False

    return True

def is_Z_pose(xyz):
    # указательный прямой, остальные согнуты
    index_ext = finger_extended(xyz, 5, 6, 7, 8, thr=165)
    mid_curled,  _, _, _ = finger_curled(xyz, 9, 10, 11, 12)
    ring_curled, _, _, _ = finger_curled(xyz, 13, 14, 15, 16)
    pink_curled, _, _, _ = finger_curled(xyz, 17, 18, 19, 20)
    thumb_not_out = ndist(xyz, 4, 0) < 1.20
    return index_ext and mid_curled and ring_curled and pink_curled and thumb_not_out

def update_Z_traj(z_traj, xyz, track_point=8):
    z_traj.append((xyz[track_point][0], xyz[track_point][1]))

def _resample_by_arclen(pts, n=72):
    """Ресемплинг траектории по длине дуги (устойчиво к скорости рисования)."""
    if len(pts) < 2:
        return pts

    # длины сегментов
    seg = []
    total = 0.0
    for i in range(1, len(pts)):
        x1, y1 = pts[i-1]
        x2, y2 = pts[i]
        d = math.hypot(x2-x1, y2-y1)
        seg.append(d)
        total += d

    if total < 1e-9:
        return [pts[0]] * n

    step = total / (n - 1)
    out = [pts[0]]
    dist_acc = 0.0
    i = 1
    cur = pts[0]

    target = step
    while len(out) < n and i < len(pts):
        prev = pts[i-1]
        nxt = pts[i]
        d = math.hypot(nxt[0]-prev[0], nxt[1]-prev[1])

        if d < 1e-9:
            i += 1
            continue

        while dist_acc + d >= target and len(out) < n:
            t = (target - dist_acc) / d
            x = prev[0] + t * (nxt[0] - prev[0])
            y = prev[1] + t * (nxt[1] - prev[1])
            out.append((x, y))
            target += step

        dist_acc += d
        i += 1

    # добиваем последней точкой
    while len(out) < n:
        out.append(pts[-1])

    return out


def _smooth3(pts):
    """Простое сглаживание (скользящее среднее 3)."""
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts)-1):
        x = (pts[i-1][0] + pts[i][0] + pts[i+1][0]) / 3.0
        y = (pts[i-1][1] + pts[i][1] + pts[i+1][1]) / 3.0
        out.append((x, y))
    out.append(pts[-1])
    return out


def is_letter_Z(z_traj, min_points=30):
    """
    Русская "З" (силуэт как '3'):
    - рисование сверху вниз (y_end > y_start)
    - 2 выраженных пика по X (две правые выпуклости)
    - между ними выраженная впадина по X
    - без резких углов (не латинская Z)
    """
    if len(z_traj) < min_points:
        return False

    pts = list(z_traj)

    # 1) нормализация по bbox (чтобы не зависеть от амплитуды)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = (maxx - minx) + 1e-9
    h = (maxy - miny) + 1e-9
    s = max(w, h)

    # минимальный “размах” движения
    if (w / s) < 0.30 or (h / s) < 0.30:
        return False

    npts = [((x - minx) / s, (y - miny) / s) for x, y in pts]

    # 2) ресемплинг + сглаживание
    npts = _resample_by_arclen(npts, n=72)
    npts = _smooth3(npts)

    xs = [p[0] for p in npts]
    ys = [p[1] for p in npts]

    # 3) движение сверху вниз (в кадре Y растёт вниз)
    if not (ys[-1] - ys[0] > 0.35):
        return False

    # 4) локальные максимумы/минимумы по X
    max_idx = []
    min_idx = []
    for i in range(1, len(xs)-1):
        if xs[i-1] < xs[i] > xs[i+1]:
            max_idx.append(i)
        if xs[i-1] > xs[i] < xs[i+1]:
            min_idx.append(i)

    if len(max_idx) < 2 or len(min_idx) < 1:
        return False

    # 5) выбираем 2 сильных “пика вправо”
    # пик должен быть заметно правее среднего
    meanx = sum(xs) / len(xs)
    strong_peaks = [i for i in max_idx if xs[i] > meanx + 0.10]
    if len(strong_peaks) < 2:
        return False

    # берём два самых правых пика, но разнесённых по времени
    strong_peaks.sort(key=lambda i: xs[i], reverse=True)
    p1 = strong_peaks[0]
    p2 = None
    for i in strong_peaks[1:]:
        if abs(i - p1) >= 12:  # разнос по траектории
            p2 = i
            break
    if p2 is None:
        return False

    # упорядочим по времени
    if p2 < p1:
        p1, p2 = p2, p1

    # 6) должна быть “впадина” между пиками (серединка "3")
    between_mins = [i for i in min_idx if p1 < i < p2]
    if not between_mins:
        return False
    valley_i = min(between_mins, key=lambda i: xs[i])

    # впадина должна быть существенно левее пиков
    peak_x = min(xs[p1], xs[p2])
    if not (peak_x - xs[valley_i] > 0.14):
        return False

    # 7) анти-латинская Z: не должно быть 2 резких углов (~ломаная из 3 сегментов)
    # считаем “скачки” направления
    angles = []
    for i in range(1, len(npts)):
        dx = npts[i][0] - npts[i-1][0]
        dy = npts[i][1] - npts[i-1][1]
        if math.hypot(dx, dy) < 1e-6:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))

    # unwrap + считаем резкие скачки
    sharp = 0
    for i in range(1, len(angles)):
        da = angles[i] - angles[i-1]
        while da > 180:
            da -= 360
        while da < -180:
            da += 360
        if abs(da) > 85:  # резкий “угол”
            sharp += 1

    # для "З" обычно максимум 0–2, для латинской Z часто 2+ стабильно
    if sharp >= 3:
        return False

    return True

# =========================
# ЦИФРЫ 1..5 (СТАТИЧЕСКИЕ)
# =========================
def get_fingers_state(xyz):
    """
    Возвращает список из 5 флагов:
    [thumb, index, middle, ring, pinky], где True = палец поднят/выпрямлен.
    Работает без handedness (лев./прав. рука) — по углам.
    """

    # Большой палец:
    #  - по углам прямой
    #  - и действительно "наружу" (не прижат к ладони/кулаку)
    thumb_ext = finger_extended(xyz, 1, 2, 3, 4, thr=150)
    thumb_out = (ndist(xyz, 4, 0) > 0.85) and (ndist(xyz, 4, 5) > 0.45)
    thumb = thumb_ext and thumb_out

    # Остальные — по углам
    index  = finger_extended(xyz, 5, 6, 7, 8,  thr=165)
    middle = finger_extended(xyz, 9, 10, 11, 12, thr=165)
    ring   = finger_extended(xyz, 13, 14, 15, 16, thr=165)
    pinky  = finger_extended(xyz, 17, 18, 19, 20, thr=165)

    return [thumb, index, middle, ring, pinky]


def is_letter_1(fingers):
    # только указательный
    return fingers == [False, True,  False, False, False]

def is_letter_2(fingers):
    # указательный + средний
    return fingers == [False, True,  True,  False, False]

def is_letter_3(fingers):
    # указательный + средний + безымянный
    return fingers == [False, True,  True,  True,  False]

def is_letter_4(fingers):
    # четыре пальца без большого
    return fingers == [False, True,  True,  True,  True]

def is_letter_5(fingers):
    # все пять
    return fingers == [True,  True,  True,  True,  True]

def detect_digit_1_5(fingers):
    """
    Возвращает строку "1".."5" или None.
    Важно: проверяем 5->1, чтобы не было ложных совпадений.
    """
    if is_letter_5(fingers): return "5"
    if is_letter_4(fingers): return "4"
    if is_letter_3(fingers): return "3"
    if is_letter_2(fingers): return "2"
    if is_letter_1(fingers): return "1"
    return None

# =========================
# СГЛАЖИВАНИЕ РЕЗУЛЬТАТА
# =========================

class LabelSmoother:
    """Храним последние метки и берём устойчивую (по большинству)."""
    def __init__(self, maxlen=HIST_LEN):
        self.hist = deque(maxlen=maxlen)

    def push(self, label_or_none):
        self.hist.append(label_or_none)

    def stable(self):
        # считаем только не-None
        vals = [x for x in self.hist if x is not None]
        if not vals:
            return None
        c = Counter(vals).most_common(1)[0]
        label, votes = c[0], c[1]
        return label if votes >= HIST_MIN_VOTES else None

# =========================
# MAIN
# =========================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Не найден hand_landmarker.task.\n"
            "Положи hand_landmarker.task рядом со скриптом (в папку проекта)."
        )

    # На Windows часто стабильнее с CAP_DSHOW
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Камера не открылась (VideoCapture(0)).")

    # Настройка MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH.resolve())),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    smoother = LabelSmoother()
    start = time.perf_counter()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # OpenCV -> RGB для MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp обязателен для VIDEO режима
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_label = None

            # Если руки найдены — рисуем и пытаемся распознать
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:

                    xyz = lms_to_xyz(hand_lms)

                    is_a = is_letter_A(hand_lms)
                    is_b = is_letter_B(hand_lms)
                    is_v = is_letter_V(hand_lms)
                    is_g = is_letter_G(hand_lms)
                    is_e = is_letter_E(hand_lms)
                    is_zh = is_letter_ZH(hand_lms)

                    # --- ЦИФРЫ 1..5 (СТАТИЧЕСКИЕ) ---
                    fingers = get_fingers_state(xyz)
                    digit = detect_digit_1_5(fingers)
                    if digit is not None:
                        detected_label = digit
                        draw_hand(frame, hand_lms, color=(0, 255, 0))
                        continue


                    # --- ДИНАМИЧЕСКАЯ БУКВА Д (САМАЯ ПЕРВАЯ) ---
                    if is_D_pose(xyz):
                        update_D_traj(d_traj, xyz)

                        if is_letter_D(d_traj):
                            detected_label = "Д"
                            draw_hand(frame, hand_lms, color=(0, 255, 0))
                            continue
                    else:
                        d_traj.clear()

                    # --- ДИНАМИЧЕСКАЯ БУКВА Ё (САМАЯ ПЕРВАЯ) ---
                    if is_YO_pose(xyz):
                        update_YO_traj(yo_traj, xyz)

                        if is_letter_YO(yo_traj):
                            detected_label = "Ё"
                            draw_hand(frame, hand_lms, color=(0, 255, 0))
                            continue
                    else:
                        yo_traj.clear()

                    # --- ДИНАМИЧЕСКАЯ БУКВА З  ---
                    if is_Z_pose(xyz):
                        update_Z_traj(z_traj, xyz, track_point=8)
                        if is_letter_Z(z_traj):
                            detected_label = "З"
                            draw_hand(frame, hand_lms, color=(0, 255, 0))
                            continue
                    else:
                        z_traj.clear()

                    # --- СТАТИЧЕСКИЕ БУКВЫ ---
                    if is_g:
                        detected_label = "Г"
                        draw_hand(frame, hand_lms, color=(0, 255, 0))

                    elif is_v:
                        detected_label = "В"
                        draw_hand(frame, hand_lms, color=(0, 255, 0))

                    elif is_b:
                        detected_label = "Б"
                        draw_hand(frame, hand_lms, color=(0, 255, 0))

                    elif is_a:
                        detected_label = "А"
                        draw_hand(frame, hand_lms, color=(0, 255, 0))
                    
                    elif is_e:
                         detected_label = "Е"
                         draw_hand(frame, hand_lms, color=(0, 255, 0))

                    if is_zh:
                        detected_label = "Ж"
                        draw_hand(frame, hand_lms, color=(0, 255, 0))

                    else:
                        draw_hand(frame, hand_lms, color=(0, 180, 255))

            # Сглаживаем метку (чтобы не мигало)
            smoother.push(detected_label)
            stable = smoother.stable()

            # UI: вывод статуса
            cv2.putText(frame, "q/ESC to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"RAW: {detected_label}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            raw_txt = detected_label if detected_label is not None else "-"

            raw_txt = detected_label if detected_label is not None else "-"
            stable_txt = stable if stable is not None else "-"

            frame = put_text_ru(frame, f"DETECTED: {raw_txt}", (10, 110), bgr=(0, 200, 255))
            frame = put_text_ru(frame, f"STABLE: {stable_txt}", (10, 155), bgr=(0, 255, 0))

            cv2.imshow("RSL (prototype)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q или ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
