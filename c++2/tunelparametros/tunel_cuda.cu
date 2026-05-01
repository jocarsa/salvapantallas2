// =========================================================
// tunel_cuda.cu
// =========================================================

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>

// =========================================================
// LIMITES INTERNOS
// =========================================================

#define MAX_NUM_LADOS 256

// =========================================================
// PARAMETROS
// =========================================================

struct Params {
    int width = 1920;
    int height = 1080;
    int fps = 60;
    int duracion = 3600;

    int num_lados = 64;
    int max_capas = 260;

    float velocidad_expansion = 1.045f;
    float radio_inicial = 8.0f;

    float noise_strength = 0.22f;
    float noise_scale = 1.7f;
    int noise_octaves = 4;

    float light_radius = 460.0f;
    float light_falloff_power = 2.6f;
    float light_intensity = 360.0f;
    float specular_intensity = 280.0f;

    float light_angle_speed = 0.014f;
    float light_radial_base = 0.56f;
    float light_radial_amp = 0.18f;
    float light_depth_base = 0.50f;
    float light_depth_amp = 0.32f;
    float light_radial_freq = 0.65f;
    float light_depth_freq = 0.27f;

    float tunnel_rotation_speed = 0.012f;
    float movement_randomness = 1.2f;
    float movement_damping = 0.97f;

    float margin_x = 420.0f;
    float margin_y = 260.0f;
    float return_force = 1.8f;

    int preview_frame = 100;
};

Params params;

// =========================================================

struct Vec2 {
    float x, y;
};

struct Vec3 {
    float x, y, z;
};

struct Capa {
    float x, y;
    float radio;
    float angulo;
    float noise[MAX_NUM_LADOS];
};

struct FaceGPU {
    float x1, y1;
    float x2, y2;
    float x3, y3;
    float x4, y4;

    unsigned char shade;

    int depthPriority;
};

// =========================================================
// GLOBAL STATE
// =========================================================

float centrox = 0.0f;
float centroy = 0.0f;

std::vector<Capa> capas;

float cursorx = 0.0f;
float cursory = 0.0f;

float vx = 0.0f;
float vy = 0.0f;

float angulo_global = 0.0f;

int ring_id_counter = 0;

// =========================================================
// LIGHT STATE
// =========================================================

float light_angle = 0.0f;

float light_radial = 0.58f;
float light_depth = 0.52f;

float light_screen_x = 0.0f;
float light_screen_y = 0.0f;

float radio_luz_visual = 10.0f;

// =========================================================
// CUDA MEMORY
// =========================================================

unsigned char* d_frame = nullptr;
int* d_depth = nullptr;
FaceGPU* d_faces = nullptr;

// =========================================================
// UTILS
// =========================================================

float clampf(float v, float a, float b) {
    return std::max(a, std::min(b, v));
}

Vec3 make_vec3(float x, float y, float z) {
    return {x, y, z};
}

Vec3 sub3(Vec3 a, Vec3 b) {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

Vec3 cross3(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float dot3(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 normalize3(Vec3 v) {
    float n = std::sqrt(dot3(v, v));

    if (n < 0.000001f) {
        return {0.0f, 0.0f, 0.0f};
    }

    return {
        v.x / n,
        v.y / n,
        v.z / n
    };
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");

    if (a == std::string::npos) {
        return "";
    }

    return s.substr(a, b - a + 1);
}

// =========================================================
// LOAD PARAMS
// =========================================================

void cargar_params(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr
            << "No se pudo abrir el archivo de parametros: "
            << filename
            << std::endl;

        std::cerr
            << "Se usaran los parametros por defecto."
            << std::endl;

        return;
    }

    std::string line;

    while (std::getline(file, line)) {
        line = trim(line);

        if (line.empty()) continue;
        if (line[0] == '#') continue;

        size_t pos = line.find('=');

        if (pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        auto to_i = [&]() {
            return std::stoi(value);
        };

        auto to_f = [&]() {
            return std::stof(value);
        };

        try {
            if (key == "width") params.width = to_i();
            else if (key == "height") params.height = to_i();
            else if (key == "fps") params.fps = to_i();
            else if (key == "duracion") params.duracion = to_i();

            else if (key == "num_lados") params.num_lados = to_i();
            else if (key == "max_capas") params.max_capas = to_i();

            else if (key == "velocidad_expansion") params.velocidad_expansion = to_f();
            else if (key == "radio_inicial") params.radio_inicial = to_f();

            else if (key == "noise_strength") params.noise_strength = to_f();
            else if (key == "noise_scale") params.noise_scale = to_f();
            else if (key == "noise_octaves") params.noise_octaves = to_i();

            else if (key == "light_radius") params.light_radius = to_f();
            else if (key == "light_falloff_power") params.light_falloff_power = to_f();
            else if (key == "light_intensity") params.light_intensity = to_f();
            else if (key == "specular_intensity") params.specular_intensity = to_f();

            else if (key == "light_angle_speed") params.light_angle_speed = to_f();
            else if (key == "light_radial_base") params.light_radial_base = to_f();
            else if (key == "light_radial_amp") params.light_radial_amp = to_f();
            else if (key == "light_depth_base") params.light_depth_base = to_f();
            else if (key == "light_depth_amp") params.light_depth_amp = to_f();
            else if (key == "light_radial_freq") params.light_radial_freq = to_f();
            else if (key == "light_depth_freq") params.light_depth_freq = to_f();

            else if (key == "tunnel_rotation_speed") params.tunnel_rotation_speed = to_f();
            else if (key == "movement_randomness") params.movement_randomness = to_f();
            else if (key == "movement_damping") params.movement_damping = to_f();

            else if (key == "margin_x") params.margin_x = to_f();
            else if (key == "margin_y") params.margin_y = to_f();
            else if (key == "return_force") params.return_force = to_f();

            else if (key == "preview_frame") params.preview_frame = to_i();
        }
        catch (...) {
            std::cerr
                << "Parametro ignorado por valor invalido: "
                << key
                << "="
                << value
                << std::endl;
        }
    }

    params.width = std::max(64, params.width);
    params.height = std::max(64, params.height);
    params.fps = std::max(1, params.fps);
    params.duracion = std::max(1, params.duracion);

    params.num_lados = std::max(3, std::min(params.num_lados, MAX_NUM_LADOS));
    params.max_capas = std::max(2, params.max_capas);

    params.noise_octaves = std::max(1, std::min(params.noise_octaves, 8));
    params.preview_frame = std::max(0, params.preview_frame);
}

// =========================================================
// FRACTAL NOISE
// =========================================================

float hash_noise(float x, float y, float z) {
    float n = std::sin(
        x * 12.9898f +
        y * 78.233f +
        z * 37.719f
    ) * 43758.5453f;

    return n - std::floor(n);
}

float fractal_noise(float x, float y, float z) {
    float total = 0.0f;
    float amp = 1.0f;
    float freq = 1.0f;
    float norm = 0.0f;

    for (int i = 0; i < params.noise_octaves; i++) {
        total += hash_noise(
            x * freq,
            y * freq,
            z * freq
        ) * amp;

        norm += amp;

        amp *= 0.5f;
        freq *= 2.0f;
    }

    return total / norm;
}

// =========================================================
// RING NOISE
// =========================================================

void crear_ruido_capa(Capa& capa, int ring_id) {
    for (int j = 0; j < params.num_lados; j++) {
        float a =
            (float(j) / float(params.num_lados)) *
            2.0f *
            float(M_PI);

        float x = std::cos(a) * params.noise_scale;
        float y = std::sin(a) * params.noise_scale;
        float z = ring_id * 0.08f;

        float n = fractal_noise(x, y, z);

        capa.noise[j] =
            (n - 0.5f) *
            params.noise_strength;
    }
}

// =========================================================
// RING POINT
// =========================================================

Vec2 punto_capa(const Capa& capa, int i) {
    float a =
        (float(i) / float(params.num_lados)) *
        2.0f *
        float(M_PI) +
        capa.angulo;

    float r =
        capa.radio *
        (1.0f + capa.noise[i]);

    return {
        capa.x + std::cos(a) * r,
        capa.y + std::sin(a) * r
    };
}

// =========================================================
// LIGHT UPDATE
// =========================================================

void actualizar_luz(float t) {
    light_angle += params.light_angle_speed;

    light_radial =
        params.light_radial_base +
        params.light_radial_amp *
        std::sin(t * params.light_radial_freq);

    light_depth =
        params.light_depth_base +
        params.light_depth_amp *
        std::sin(t * params.light_depth_freq + 1.2f);

    light_radial =
        clampf(light_radial, 0.05f, 0.95f);

    light_depth =
        clampf(light_depth, 0.05f, 0.95f);

    if (capas.size() < 2) {
        light_screen_x = centrox;
        light_screen_y = centroy;
        return;
    }

    float pos =
        light_depth *
        float(capas.size() - 1);

    int idx0 = int(std::floor(pos));
    int idx1 = std::min(
        idx0 + 1,
        int(capas.size() - 1)
    );

    float ti = pos - float(idx0);

    Capa& c0 = capas[idx0];
    Capa& c1 = capas[idx1];

    float cx =
        c0.x * (1.0f - ti) +
        c1.x * ti;

    float cy =
        c0.y * (1.0f - ti) +
        c1.y * ti;

    float radio =
        c0.radio * (1.0f - ti) +
        c1.radio * ti;

    float angulo =
        c0.angulo * (1.0f - ti) +
        c1.angulo * ti;

    float usable_radius =
        radio *
        light_radial;

    light_screen_x =
        cx +
        std::cos(light_angle + angulo) *
        usable_radius;

    light_screen_y =
        cy +
        std::sin(light_angle + angulo) *
        usable_radius;
}

// =========================================================
// FACE GENERATION
// =========================================================

void construir_faces(
    std::vector<FaceGPU>& faces
) {
    faces.clear();

    if (capas.size() < 2) {
        return;
    }

    float zscale = 1200.0f;

    Vec3 light_pos = make_vec3(
        light_screen_x,
        light_screen_y,
        light_depth * zscale
    );

    int ncapas = int(capas.size());

    for (int i = ncapas - 1; i > 0; i--) {
        Capa& capa_ext = capas[i];
        Capa& capa_int = capas[i - 1];

        float depth_ext =
            float(i) /
            float(ncapas - 1);

        float depth_int =
            float(i - 1) /
            float(ncapas - 1);

        for (int j = 0; j < params.num_lados; j++) {
            int j2 =
                (j + 1) %
                params.num_lados;

            Vec2 p1 = punto_capa(capa_ext, j);
            Vec2 p2 = punto_capa(capa_ext, j2);

            Vec2 p3 = punto_capa(capa_int, j2);
            Vec2 p4 = punto_capa(capa_int, j);

            Vec2 center = {
                (p1.x + p2.x + p3.x + p4.x) * 0.25f,
                (p1.y + p2.y + p3.y + p4.y) * 0.25f
            };

            float face_depth =
                (depth_ext + depth_int) * 0.5f;

            Vec3 face_pos = make_vec3(
                center.x,
                center.y,
                face_depth * zscale
            );

            Vec3 v1 = make_vec3(
                p2.x - p1.x,
                p2.y - p1.y,
                0.0f
            );

            Vec3 v2 = make_vec3(
                p4.x - p1.x,
                p4.y - p1.y,
                (depth_int - depth_ext) * zscale
            );

            Vec3 normal =
                normalize3(
                    cross3(v1, v2)
                );

            Vec3 light_dir =
                normalize3(
                    sub3(light_pos, face_pos)
                );

            float diffuse =
                std::max(
                    0.0f,
                    dot3(normal, light_dir)
                );

            float dx =
                light_pos.x - face_pos.x;

            float dy =
                light_pos.y - face_pos.y;

            float dz =
                light_pos.z - face_pos.z;

            float dist =
                std::sqrt(
                    dx * dx +
                    dy * dy +
                    dz * dz
                );

            float attenuation =
                1.0f -
                dist / params.light_radius;

            attenuation =
                clampf(
                    attenuation,
                    0.0f,
                    1.0f
                );

            attenuation =
                std::pow(
                    attenuation,
                    params.light_falloff_power
                );

            Vec3 view_dir =
                make_vec3(
                    0.0f,
                    0.0f,
                    -1.0f
                );

            float ndotl =
                dot3(normal, light_dir);

            Vec3 reflect_dir =
                normalize3({
                    2.0f * ndotl * normal.x - light_dir.x,
                    2.0f * ndotl * normal.y - light_dir.y,
                    2.0f * ndotl * normal.z - light_dir.z
                });

            float specular =
                std::pow(
                    std::max(
                        0.0f,
                        dot3(view_dir, reflect_dir)
                    ),
                    28.0f
                );

            float ambient =
                6.0f +
                face_depth * 16.0f;

            float light =
                diffuse *
                attenuation *
                params.light_intensity +

                specular *
                attenuation *
                params.specular_intensity;

            int shade =
                int(
                    clampf(
                        ambient + light,
                        2.0f,
                        240.0f
                    )
                );

            FaceGPU f;

            f.x1 = p1.x;
            f.y1 = p1.y;

            f.x2 = p2.x;
            f.y2 = p2.y;

            f.x3 = p3.x;
            f.y3 = p3.y;

            f.x4 = p4.x;
            f.y4 = p4.y;

            f.shade =
                (unsigned char)shade;

            f.depthPriority =
                int(face_depth * 1000000.0f);

            faces.push_back(f);
        }
    }
}

// =========================================================
// CUDA
// =========================================================

__device__ float edge_func(
    float ax,
    float ay,
    float bx,
    float by,
    float px,
    float py
) {
    return
        (px - ax) * (by - ay) -
        (py - ay) * (bx - ax);
}

__device__ bool point_in_tri(
    float px,
    float py,

    float ax,
    float ay,

    float bx,
    float by,

    float cx,
    float cy
) {
    float e1 =
        edge_func(
            ax, ay,
            bx, by,
            px, py
        );

    float e2 =
        edge_func(
            bx, by,
            cx, cy,
            px, py
        );

    float e3 =
        edge_func(
            cx, cy,
            ax, ay,
            px, py
        );

    bool has_neg =
        (e1 < 0.0f) ||
        (e2 < 0.0f) ||
        (e3 < 0.0f);

    bool has_pos =
        (e1 > 0.0f) ||
        (e2 > 0.0f) ||
        (e3 > 0.0f);

    return !(has_neg && has_pos);
}

__global__ void clear_kernel(
    unsigned char* frame,
    int* depth,
    int width,
    int height
) {
    int idx =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int total =
        width * height;

    if (idx >= total) {
        return;
    }

    frame[idx * 3 + 0] = 0;
    frame[idx * 3 + 1] = 0;
    frame[idx * 3 + 2] = 0;

    depth[idx] =
        -2147483647;
}

__global__ void raster_faces_kernel(
    unsigned char* frame,
    int* depth,
    FaceGPU* faces,
    int face_count,
    int width,
    int height
) {
    int face_id = blockIdx.x;

    if (face_id >= face_count) {
        return;
    }

    FaceGPU f = faces[face_id];

    float minx =
        fminf(
            fminf(f.x1, f.x2),
            fminf(f.x3, f.x4)
        );

    float maxx =
        fmaxf(
            fmaxf(f.x1, f.x2),
            fmaxf(f.x3, f.x4)
        );

    float miny =
        fminf(
            fminf(f.y1, f.y2),
            fminf(f.y3, f.y4)
        );

    float maxy =
        fmaxf(
            fmaxf(f.y1, f.y2),
            fmaxf(f.y3, f.y4)
        );

    int ix0 =
        max(0, int(floorf(minx)));

    int iy0 =
        max(0, int(floorf(miny)));

    int ix1 =
        min(width - 1, int(ceilf(maxx)));

    int iy1 =
        min(height - 1, int(ceilf(maxy)));

    int bw =
        ix1 - ix0 + 1;

    int bh =
        iy1 - iy0 + 1;

    if (bw <= 0 || bh <= 0) {
        return;
    }

    int total = bw * bh;

    int local =
        threadIdx.x;

    for (
        int k = local;
        k < total;
        k += blockDim.x
    ) {
        int lx = k % bw;
        int ly = k / bw;

        int x = ix0 + lx;
        int y = iy0 + ly;

        float px = float(x) + 0.5f;
        float py = float(y) + 0.5f;

        bool inside =
            point_in_tri(
                px, py,
                f.x1, f.y1,
                f.x2, f.y2,
                f.x3, f.y3
            ) ||

            point_in_tri(
                px, py,
                f.x1, f.y1,
                f.x3, f.y3,
                f.x4, f.y4
            );

        if (!inside) {
            continue;
        }

        int pixel =
            y * width + x;

        int old =
            atomicMax(
                &depth[pixel],
                f.depthPriority
            );

        if (f.depthPriority >= old) {
            frame[pixel * 3 + 0] =
                f.shade;

            frame[pixel * 3 + 1] =
                f.shade;

            frame[pixel * 3 + 2] =
                f.shade;
        }
    }
}

// =========================================================
// CPU WIREFRAME
// =========================================================

void dibujar_wireframe_cpu(
    cv::Mat& img,
    const std::vector<FaceGPU>& faces
) {
    for (const auto& f : faces) {
        int edge =
            int(f.shade) + 45;

        edge =
            int(
                clampf(
                    float(edge),
                    25.0f,
                    255.0f
                )
            );

        cv::Scalar color(
            edge,
            edge,
            edge
        );

        cv::Point p1(
            int(f.x1),
            int(f.y1)
        );

        cv::Point p2(
            int(f.x2),
            int(f.y2)
        );

        cv::Point p3(
            int(f.x3),
            int(f.y3)
        );

        cv::Point p4(
            int(f.x4),
            int(f.y4)
        );

        cv::line(
            img,
            p1,
            p2,
            color,
            1,
            cv::LINE_AA
        );

        cv::line(
            img,
            p2,
            p3,
            color,
            1,
            cv::LINE_AA
        );

        cv::line(
            img,
            p3,
            p4,
            color,
            1,
            cv::LINE_AA
        );

        cv::line(
            img,
            p4,
            p1,
            color,
            1,
            cv::LINE_AA
        );
    }
}

// =========================================================
// DRAW LIGHT
// =========================================================

void dibujar_luz_cpu(cv::Mat& img) {
    float perspective_size =
        1.60f -
        light_depth * 1.25f;

    int radius =
        int(
            radio_luz_visual *
            perspective_size
        );

    radius =
        int(
            clampf(
                float(radius),
                3.0f,
                28.0f
            )
        );

    cv::circle(
        img,
        cv::Point(
            int(light_screen_x),
            int(light_screen_y)
        ),
        radius,
        cv::Scalar(255, 255, 255),
        -1,
        cv::LINE_AA
    );
}

// =========================================================
// SIMULATION STEP
// =========================================================

void simular_paso() {
    vx +=
        ((float(std::rand()) / RAND_MAX) - 0.5f) *
        params.movement_randomness;

    vy +=
        ((float(std::rand()) / RAND_MAX) - 0.5f) *
        params.movement_randomness;

    vx *= params.movement_damping;
    vy *= params.movement_damping;

    cursorx += vx;
    cursory += vy;

    if (cursorx < centrox - params.margin_x)
        vx += params.return_force;

    if (cursorx > centrox + params.margin_x)
        vx -= params.return_force;

    if (cursory < centroy - params.margin_y)
        vy += params.return_force;

    if (cursory > centroy + params.margin_y)
        vy -= params.return_force;

    angulo_global += params.tunnel_rotation_speed;

    Capa nueva;

    nueva.x = cursorx;
    nueva.y = cursory;
    nueva.radio = params.radio_inicial;
    nueva.angulo = angulo_global;

    crear_ruido_capa(
        nueva,
        ring_id_counter
    );

    ring_id_counter++;

    capas.push_back(nueva);

    if ((int)capas.size() > params.max_capas) {
        capas.erase(capas.begin());
    }

    for (auto& c : capas) {
        c.radio *= params.velocidad_expansion;
    }

    capas.erase(
        std::remove_if(
            capas.begin(),
            capas.end(),
            [](const Capa& c) {
                return c.radio >= params.width * 1.6f;
            }
        ),
        capas.end()
    );
}

// =========================================================
// MAIN
// =========================================================

int main(int argc, char** argv) {
    std::srand(
        unsigned(std::time(nullptr))
    );

    std::string params_file =
        "params_tunel.txt";

    bool preview_mode =
        false;

    for (int i = 1; i < argc; i++) {
        std::string arg =
            argv[i];

        if (arg == "--params" && i + 1 < argc) {
            params_file =
                argv[i + 1];

            i++;
        }
        else if (arg == "--preview") {
            preview_mode =
                true;
        }
    }

    cargar_params(params_file);

    centrox = params.width * 0.5f;
    centroy = params.height * 0.5f;

    cursorx = centrox;
    cursory = centroy;

    light_screen_x = centrox;
    light_screen_y = centroy;

    int max_faces =
        params.max_capas *
        params.num_lados;

    size_t frame_bytes =
        size_t(params.width) *
        size_t(params.height) *
        3;

    size_t depth_bytes =
        size_t(params.width) *
        size_t(params.height) *
        sizeof(int);

    cudaMalloc(
        &d_frame,
        frame_bytes
    );

    cudaMalloc(
        &d_depth,
        depth_bytes
    );

    cudaMalloc(
        &d_faces,
        max_faces * sizeof(FaceGPU)
    );

    cv::namedWindow(
        preview_mode ? "Preview" : "Frame",
        cv::WINDOW_NORMAL
    );

    cv::resizeWindow(
        preview_mode ? "Preview" : "Frame",
        params.width,
        params.height
    );

    std::time_t epoch =
        std::time(nullptr);

    std::string filename =
        "tunnel_cuda_" +
        std::to_string(epoch) +
        ".mp4";

    cv::VideoWriter out;

    if (!preview_mode) {
        out.open(
            filename,
            cv::VideoWriter::fourcc(
                'm',
                'p',
                '4',
                'v'
            ),
            params.fps,
            cv::Size(
                params.width,
                params.height
            )
        );

        if (!out.isOpened()) {
            std::cerr
                << "No se ha podido abrir el VideoWriter."
                << std::endl;

            cudaFree(d_frame);
            cudaFree(d_depth);
            cudaFree(d_faces);

            return 1;
        }
    }

    cv::Mat frame(
        params.height,
        params.width,
        CV_8UC3
    );

    cv::Mat blur;

    std::vector<FaceGPU> faces;

    faces.reserve(max_faces);

    int total_frames =
        params.fps *
        params.duracion;

    if (preview_mode) {
        total_frames =
            params.preview_frame + 1;
    }

    auto start =
        std::chrono::high_resolution_clock::now();

    for (
        int contador = 0;
        contador < total_frames;
        contador++
    ) {
        float t =
            float(contador) /
            float(params.fps);

        simular_paso();

        actualizar_luz(t);

        construir_faces(faces);

        int total_pixels =
            params.width *
            params.height;

        int threads_clear =
            256;

        int blocks_clear =
            (total_pixels + threads_clear - 1) /
            threads_clear;

        clear_kernel<<<
            blocks_clear,
            threads_clear
        >>>(
            d_frame,
            d_depth,
            params.width,
            params.height
        );

        if (!faces.empty()) {
            cudaMemcpy(
                d_faces,
                faces.data(),
                faces.size() * sizeof(FaceGPU),
                cudaMemcpyHostToDevice
            );

            raster_faces_kernel<<<
                int(faces.size()),
                256
            >>>(
                d_frame,
                d_depth,
                d_faces,
                int(faces.size()),
                params.width,
                params.height
            );
        }

        cudaDeviceSynchronize();

        cudaMemcpy(
            frame.data,
            d_frame,
            frame_bytes,
            cudaMemcpyDeviceToHost
        );

        dibujar_wireframe_cpu(
            frame,
            faces
        );

        cv::GaussianBlur(
            frame,
            blur,
            cv::Size(0, 0),
            6
        );

        cv::addWeighted(
            frame,
            1.0,
            blur,
            0.06,
            0.0,
            frame
        );

        dibujar_luz_cpu(frame);

        if (preview_mode && contador == params.preview_frame) {
            cv::imwrite(
                "preview_tunel.png",
                frame
            );

            cv::imshow(
                "Preview",
                frame
            );

            std::cout
                << "Preview guardado en: preview_tunel.png"
                << std::endl;

            cv::waitKey(0);
            break;
        }

        if (!preview_mode) {
            cv::imshow(
                "Frame",
                frame
            );

            out.write(frame);

            int key =
                cv::waitKey(1) & 0xFF;

            if (key == 27 || key == 'q') {
                break;
            }
        }
    }

    if (!preview_mode) {
        out.release();

        std::cout
            << "Video guardado en: "
            << filename
            << std::endl;
    }

    cudaFree(d_frame);
    cudaFree(d_depth);
    cudaFree(d_faces);

    cv::destroyAllWindows();

    return 0;
}
