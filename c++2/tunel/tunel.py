import cv2
import numpy as np
import math
import time

width = 1920
height = 1080
centrox = width // 2
centroy = height // 2

fps = 60
duracion = 60 * 60
totalframes = fps * duracion

numerodelados = 64
velocidadexpansion = 1.045
maxcapas = 260
radio_inicial = 8
grosor_linea = 1

# Surface noise
noise_strength = 0.22
noise_scale = 1.7
noise_octaves = 4

# Light falloff
light_radius = 420.0
light_falloff_power = 3.0

frame = np.zeros((height, width, 3), dtype=np.uint8)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", width, height)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_filename = f"tunnel_{int(time.time())}.mp4"
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

capas = []
ring_id_counter = 0

cursorx = float(centrox)
cursory = float(centroy)
vx = 0.0
vy = 0.0
angulo_global = 0.0

light_angle = 0.0
light_angle_speed = 0.014
light_radial = 0.58
light_depth = 0.52
light_screen_x = float(centrox)
light_screen_y = float(centroy)
radio_luz_visual = 10


def clamp(v, a, b):
    return max(a, min(b, v))


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def hash_noise(x, y, z):
    n = math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453
    return n - math.floor(n)


def fractal_noise(x, y, z):
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0

    for _ in range(noise_octaves):
        total += hash_noise(x * freq, y * freq, z * freq) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0

    return total / norm


def crear_ruido_capa(ring_id):
    valores = []

    for j in range(numerodelados):
        a = (j / numerodelados) * math.pi * 2
        x = math.cos(a) * noise_scale
        y = math.sin(a) * noise_scale
        z = ring_id * 0.08

        n = fractal_noise(x, y, z)
        valores.append((n - 0.5) * noise_strength)

    return valores


def generar_poligono(capa):
    puntos = []

    for i in range(numerodelados):
        a = (i / numerodelados) * math.pi * 2 + capa["angulo"]
        r = capa["radio"] * (1.0 + capa["noise"][i])

        x = capa["x"] + math.cos(a) * r
        y = capa["y"] + math.sin(a) * r

        puntos.append(np.array([x, y], dtype=np.float32))

    return puntos


def actualizar_luz(t):
    global light_angle, light_radial, light_depth
    global light_screen_x, light_screen_y

    light_angle += light_angle_speed

    light_radial = 0.56 + 0.18 * math.sin(t * 0.65)
    light_depth = 0.50 + 0.32 * math.sin(t * 0.27 + 1.2)

    light_radial = clamp(light_radial, 0.25, 0.82)
    light_depth = clamp(light_depth, 0.12, 0.90)

    if len(capas) < 2:
        return

    pos = light_depth * (len(capas) - 1)
    idx0 = int(math.floor(pos))
    idx1 = min(idx0 + 1, len(capas) - 1)
    t_interp = pos - idx0

    c0 = capas[idx0]
    c1 = capas[idx1]

    cx = c0["x"] * (1 - t_interp) + c1["x"] * t_interp
    cy = c0["y"] * (1 - t_interp) + c1["y"] * t_interp
    radio = c0["radio"] * (1 - t_interp) + c1["radio"] * t_interp
    angulo = c0["angulo"] * (1 - t_interp) + c1["angulo"] * t_interp

    usable_radius = radio * light_radial

    light_screen_x = cx + math.cos(light_angle + angulo) * usable_radius
    light_screen_y = cy + math.sin(light_angle + angulo) * usable_radius


def dibujar_luz(frame):
    perspective_size = 1.60 - light_depth * 1.25
    radius = int(radio_luz_visual * perspective_size)
    radius = int(clamp(radius, 3, 28))

    cv2.circle(
        frame,
        (int(light_screen_x), int(light_screen_y)),
        radius,
        (255, 255, 255),
        -1
    )


def dibujar_tunel(frame):
    global capas

    for capa in capas:
        capa["radio"] *= velocidadexpansion

    capas = [c for c in capas if c["radio"] < width * 1.6]

    if len(capas) < 2:
        return

    zscale = 1200.0

    for i in range(len(capas) - 1, 0, -1):

        capa_ext = capas[i]
        capa_int = capas[i - 1]

        depth_ext = i / max(1, len(capas) - 1)
        depth_int = (i - 1) / max(1, len(capas) - 1)

        poly_ext = generar_poligono(capa_ext)
        poly_int = generar_poligono(capa_int)

        for j in range(numerodelados):

            p1 = poly_ext[j]
            p2 = poly_ext[(j + 1) % numerodelados]
            p3 = poly_int[(j + 1) % numerodelados]
            p4 = poly_int[j]

            center = (p1 + p2 + p3 + p4) * 0.25
            face_depth = (depth_ext + depth_int) * 0.5

            face_pos = np.array([
                center[0],
                center[1],
                face_depth * zscale
            ], dtype=np.float32)

            light_pos = np.array([
                light_screen_x,
                light_screen_y,
                light_depth * zscale
            ], dtype=np.float32)

            v1 = np.array([
                p2[0] - p1[0],
                p2[1] - p1[1],
                0.0
            ], dtype=np.float32)

            v2 = np.array([
                p4[0] - p1[0],
                p4[1] - p1[1],
                (depth_int - depth_ext) * zscale
            ], dtype=np.float32)

            normal = normalize(np.cross(v1, v2))
            light_dir = normalize(light_pos - face_pos)

            diffuse = max(0.0, np.dot(normal, light_dir))

            dist = np.linalg.norm(light_pos - face_pos)

            attenuation = 1.0 - dist / light_radius
            attenuation = clamp(attenuation, 0.0, 1.0)
            attenuation = attenuation ** light_falloff_power

            view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            reflect_dir = normalize(
                2.0 * np.dot(normal, light_dir) * normal - light_dir
            )

            specular = max(0.0, np.dot(view_dir, reflect_dir)) ** 28.0

            ambient = 8 + face_depth * 24
            light = diffuse * attenuation * 520 + specular * attenuation * 900

            shade = int(clamp(ambient + light, 3, 255))
            edge = int(clamp(shade + 28, 10, 255))

            face = np.array([
                [int(p1[0]), int(p1[1])],
                [int(p2[0]), int(p2[1])],
                [int(p3[0]), int(p3[1])],
                [int(p4[0]), int(p4[1])]
            ], dtype=np.int32)

            cv2.fillConvexPoly(frame, face, (shade, shade, shade))
            cv2.polylines(frame, [face], True, (edge, edge, edge), grosor_linea)


contador = 0
inicio = time.time()

try:
    while contador < totalframes:

        contador += 1
        t = time.time() - inicio

        frame[:] = (0, 0, 0)

        vx += (np.random.rand() - 0.5) * 1.2
        vy += (np.random.rand() - 0.5) * 1.2

        vx *= 0.97
        vy *= 0.97

        cursorx += vx
        cursory += vy

        margen_x = 420
        margen_y = 260

        if cursorx < centrox - margen_x:
            vx += 1.8

        if cursorx > centrox + margen_x:
            vx -= 1.8

        if cursory < centroy - margen_y:
            vy += 1.8

        if cursory > centroy + margen_y:
            vy -= 1.8

        angulo_global += 0.012

        capas.append({
            "x": cursorx,
            "y": cursory,
            "radio": radio_inicial,
            "angulo": angulo_global,
            "noise": crear_ruido_capa(ring_id_counter)
        })

        ring_id_counter += 1

        if len(capas) > maxcapas:
            capas.pop(0)

        actualizar_luz(t)
        dibujar_tunel(frame)

        blur = cv2.GaussianBlur(frame, (0, 0), 6)
        frame = cv2.addWeighted(frame, 1.0, blur, 0.10, 0)

        dibujar_luz(frame)

        cv2.imshow("Frame", frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    out.release()
    cv2.destroyAllWindows()
    print(f"Video guardado en: {output_filename}")
