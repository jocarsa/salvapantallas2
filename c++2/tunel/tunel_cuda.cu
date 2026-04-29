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
#include <iomanip>
#include <sstream>
#include <string>

#define WIDTH 1920
#define HEIGHT 1080

#define FPS 60
#define DURACION 600
#define TOTAL_FRAMES (FPS * DURACION)

#define NUM_LADOS 64
#define MAX_CAPAS 260

#define VELOCIDAD_EXPANSION 1.045f
#define RADIO_INICIAL 8.0f

#define NOISE_STRENGTH 0.22f
#define NOISE_SCALE 10.7f
#define NOISE_OCTAVES 4

#define LIGHT_REACH 520.0f
#define LIGHT_FALLOFF_POWER 3.2f
#define LIGHT_INTENSITY 1.85f
#define SPECULAR_INTENSITY 120.0f

#define WIREFRAME_R 255
#define WIREFRAME_G 255
#define WIREFRAME_B 255
#define WIREFRAME_BOOST 127

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
    float noise[NUM_LADOS];
};

struct FaceGPU {
    float x1, y1;
    float x2, y2;
    float x3, y3;
    float x4, y4;
    unsigned char shade;
    int depthPriority;
};

float centrox = WIDTH * 0.5f;
float centroy = HEIGHT * 0.5f;

std::vector<Capa> capas;

float cursorx = centrox;
float cursory = centroy;
float vx = 0.0f;
float vy = 0.0f;
float angulo_global = 0.0f;

int ring_id_counter = 0;

float light_angle = 0.0f;
float light_angle_speed = 0.014f;
float light_radial = 0.58f;
float light_depth = 0.52f;
float light_screen_x = centrox;
float light_screen_y = centroy;
float radio_luz_visual = 10.0f;

unsigned char* d_frame = nullptr;
int* d_depth = nullptr;
FaceGPU* d_faces = nullptr;

int max_faces = MAX_CAPAS * NUM_LADOS;

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
    return {a.x - b.x, a.y - b.y, a.z - b.z};
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

    return {v.x / n, v.y / n, v.z / n};
}

std::string format_time(double seconds) {
    int s = int(seconds);

    if (s < 0) {
        s = 0;
    }

    int h = s / 3600;
    int m = (s % 3600) / 60;
    int sec = s % 60;

    std::ostringstream oss;

    if (h > 0) {
        oss
            << h << "h "
            << std::setw(2) << std::setfill('0') << m << "m "
            << std::setw(2) << std::setfill('0') << sec << "s";
    } else {
        oss
            << m << "m "
            << std::setw(2) << std::setfill('0') << sec << "s";
    }

    return oss.str();
}

void print_progress_bar(
    int current,
    int total,
    double elapsed,
    const std::string& filename
) {
    const int bar_width = 42;

    double progress = double(current) / double(total);
    progress = clampf(float(progress), 0.0f, 1.0f);

    int filled = int(progress * bar_width);

    double render_fps = current / std::max(0.001, elapsed);
    double eta = (total - current) / std::max(0.001, render_fps);

    std::cout << "\r";
    std::cout << "Renderizando ";
    std::cout << "[";

    for (int i = 0; i < bar_width; i++) {
        if (i < filled) {
            std::cout << "█";
        } else if (i == filled) {
            std::cout << "▓";
        } else {
            std::cout << "░";
        }
    }

    std::cout << "] ";

    std::cout
        << std::fixed
        << std::setprecision(1)
        << progress * 100.0
        << "%  ";

    std::cout
        << current
        << "/"
        << total
        << " frames  ";

    std::cout
        << std::setprecision(1)
        << render_fps
        << " fps  ";

    std::cout
        << "ETA "
        << format_time(eta)
        << "  ";

    std::cout
        << "-> "
        << filename
        << "      ";

    std::cout.flush();
}

// =========================================================
// NOISE
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

    for (int i = 0; i < NOISE_OCTAVES; i++) {
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

void crear_ruido_capa(Capa& capa, int ring_id) {
    for (int j = 0; j < NUM_LADOS; j++) {
        float a =
            (float(j) / float(NUM_LADOS)) *
            2.0f *
            float(M_PI);

        float x = std::cos(a) * NOISE_SCALE;
        float y = std::sin(a) * NOISE_SCALE;
        float z = ring_id * 0.08f;

        float n = fractal_noise(x, y, z);

        capa.noise[j] =
            (n - 0.5f) *
            NOISE_STRENGTH;
    }
}

Vec2 punto_capa(const Capa& capa, int i) {
    float a =
        (float(i) / float(NUM_LADOS)) *
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
// LIGHT
// =========================================================

void actualizar_luz(float t) {
    light_angle += light_angle_speed;

    light_radial =
        0.56f +
        0.18f *
        std::sin(t * 0.65f);

    light_depth =
        0.50f +
        0.32f *
        std::sin(t * 0.27f + 1.2f);

    light_radial =
        clampf(light_radial, 0.25f, 0.82f);

    light_depth =
        clampf(light_depth, 0.12f, 0.90f);

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

void construir_faces(std::vector<FaceGPU>& faces) {
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

        for (int j = 0; j < NUM_LADOS; j++) {
            int j2 =
                (j + 1) %
                NUM_LADOS;

            Vec2 p1 = punto_capa(capa_ext, j);
            Vec2 p2 = punto_capa(capa_ext, j2);
            Vec2 p3 = punto_capa(capa_int, j2);
            Vec2 p4 = punto_capa(capa_int, j);

            Vec2 center = {
                (p1.x + p2.x + p3.x + p4.x) * 0.25f,
                (p1.y + p2.y + p3.y + p4.y) * 0.25f
            };

            float face_depth =
                (depth_ext + depth_int) *
                0.5f;

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

            float dx = light_pos.x - face_pos.x;
            float dy = light_pos.y - face_pos.y;
            float dz = light_pos.z - face_pos.z;

            float dist =
                std::sqrt(
                    dx * dx +
                    dy * dy +
                    dz * dz
                );

            float normalized_dist =
                clampf(
                    dist / LIGHT_REACH,
                    0.0f,
                    1.0f
                );

            float attenuation =
                1.0f - normalized_dist;

            attenuation =
                std::pow(
                    attenuation,
                    LIGHT_FALLOFF_POWER
                );

            float inverse_square =
                1.0f /
                (
                    1.0f +
                    normalized_dist *
                    normalized_dist *
                    8.0f
                );

            attenuation *= inverse_square;

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

            float ambient = 4.0f;

            float light =
                diffuse *
                attenuation *
                255.0f *
                LIGHT_INTENSITY +

                specular *
                attenuation *
                SPECULAR_INTENSITY *
                LIGHT_INTENSITY;

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
// CUDA KERNELS
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
    int* depth
) {
    int idx =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int total =
        WIDTH * HEIGHT;

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
    int face_count
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
        max(
            0,
            int(floorf(minx))
        );

    int iy0 =
        max(
            0,
            int(floorf(miny))
        );

    int ix1 =
        min(
            WIDTH - 1,
            int(ceilf(maxx))
        );

    int iy1 =
        min(
            HEIGHT - 1,
            int(ceilf(maxy))
        );

    int bw =
        ix1 - ix0 + 1;

    int bh =
        iy1 - iy0 + 1;

    if (bw <= 0 || bh <= 0) {
        return;
    }

    int total = bw * bh;
    int local = threadIdx.x;

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
            y * WIDTH + x;

        int old =
            atomicMax(
                &depth[pixel],
                f.depthPriority
            );

        if (f.depthPriority >= old) {
            frame[pixel * 3 + 0] = f.shade;
            frame[pixel * 3 + 1] = f.shade;
            frame[pixel * 3 + 2] = f.shade;
        }
    }
}

// =========================================================
// CPU DRAWING
// =========================================================

void dibujar_wireframe_cpu(
    cv::Mat& img,
    const std::vector<FaceGPU>& faces
) {
    for (const auto& f : faces) {
        float factor =
            clampf(
                (float(f.shade) + float(WIREFRAME_BOOST)) / 255.0f,
                0.0f,
                1.0f
            );

        cv::Scalar color(
            int(WIREFRAME_B * factor),
            int(WIREFRAME_G * factor),
            int(WIREFRAME_R * factor)
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
// MAIN
// =========================================================

int main() {
    std::srand(
        unsigned(std::time(nullptr))
    );

    cudaMalloc(
        &d_frame,
        WIDTH * HEIGHT * 3
    );

    cudaMalloc(
        &d_depth,
        WIDTH * HEIGHT * sizeof(int)
    );

    cudaMalloc(
        &d_faces,
        max_faces * sizeof(FaceGPU)
    );

    cv::namedWindow(
        "Frame",
        cv::WINDOW_NORMAL
    );

    cv::resizeWindow(
        "Frame",
        WIDTH,
        HEIGHT
    );

    std::time_t epoch =
        std::time(nullptr);

    std::string filename =
        "tunnel_cuda_" +
        std::to_string(epoch) +
        ".mp4";

    cv::VideoWriter out(
        filename,
        cv::VideoWriter::fourcc(
            'm',
            'p',
            '4',
            'v'
        ),
        FPS,
        cv::Size(
            WIDTH,
            HEIGHT
        )
    );

    if (!out.isOpened()) {
        std::cerr
            << "No se ha podido abrir el VideoWriter."
            << std::endl;

        return 1;
    }

    cv::Mat frame(
        HEIGHT,
        WIDTH,
        CV_8UC3
    );

    cv::Mat blur;

    std::vector<FaceGPU> faces;

    faces.reserve(max_faces);

    auto start =
        std::chrono::high_resolution_clock::now();

    std::cout
        << "Generando vídeo: "
        << filename
        << std::endl;

    for (
        int contador = 0;
        contador < TOTAL_FRAMES;
        contador++
    ) {
        auto now =
            std::chrono::high_resolution_clock::now();

        float t =
            std::chrono::duration<float>(
                now - start
            ).count();

        // -------------------------------------------------
        // MOVEMENT
        // -------------------------------------------------

        vx +=
            ((float(std::rand()) / RAND_MAX) - 0.5f) *
            1.2f;

        vy +=
            ((float(std::rand()) / RAND_MAX) - 0.5f) *
            1.2f;

        vx *= 0.97f;
        vy *= 0.97f;

        cursorx += vx;
        cursory += vy;

        float margen_x = 420.0f;
        float margen_y = 260.0f;

        if (cursorx < centrox - margen_x) {
            vx += 1.8f;
        }

        if (cursorx > centrox + margen_x) {
            vx -= 1.8f;
        }

        if (cursory < centroy - margen_y) {
            vy += 1.8f;
        }

        if (cursory > centroy + margen_y) {
            vy -= 1.8f;
        }

        angulo_global += 0.012f;

        // -------------------------------------------------
        // CREATE RING
        // -------------------------------------------------

        Capa nueva;

        nueva.x = cursorx;
        nueva.y = cursory;
        nueva.radio = RADIO_INICIAL;
        nueva.angulo = angulo_global;

        crear_ruido_capa(
            nueva,
            ring_id_counter
        );

        ring_id_counter++;

        capas.push_back(nueva);

        if ((int)capas.size() > MAX_CAPAS) {
            capas.erase(capas.begin());
        }

        // -------------------------------------------------
        // EXPAND
        // -------------------------------------------------

        for (auto& c : capas) {
            c.radio *= VELOCIDAD_EXPANSION;
        }

        capas.erase(
            std::remove_if(
                capas.begin(),
                capas.end(),
                [](const Capa& c) {
                    return c.radio >= WIDTH * 1.6f;
                }
            ),
            capas.end()
        );

        // -------------------------------------------------
        // LIGHT + FACES
        // -------------------------------------------------

        actualizar_luz(t);

        construir_faces(faces);

        // -------------------------------------------------
        // CLEAR GPU FRAME
        // -------------------------------------------------

        int total_pixels =
            WIDTH * HEIGHT;

        int threads_clear = 256;

        int blocks_clear =
            (total_pixels + threads_clear - 1) /
            threads_clear;

        clear_kernel<<<
            blocks_clear,
            threads_clear
        >>>(
            d_frame,
            d_depth
        );

        // -------------------------------------------------
        // RASTER GPU
        // -------------------------------------------------

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
                int(faces.size())
            );
        }

        cudaDeviceSynchronize();

        // -------------------------------------------------
        // COPY BACK
        // -------------------------------------------------

        cudaMemcpy(
            frame.data,
            d_frame,
            WIDTH * HEIGHT * 3,
            cudaMemcpyDeviceToHost
        );

        // -------------------------------------------------
        // WIREFRAME
        // -------------------------------------------------

        dibujar_wireframe_cpu(
            frame,
            faces
        );

        // -------------------------------------------------
        // BLOOM
        // -------------------------------------------------

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

        // -------------------------------------------------
        // LIGHT
        // -------------------------------------------------

        dibujar_luz_cpu(frame);

        // -------------------------------------------------
        // DISPLAY / SAVE
        // -------------------------------------------------

        cv::imshow(
            "Frame",
            frame
        );

        out.write(frame);

        // -------------------------------------------------
        // PROGRESS BAR
        // -------------------------------------------------

        auto progress_now =
            std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double>(
                progress_now - start
            ).count();

        if (
            contador % 10 == 0 ||
            contador == TOTAL_FRAMES - 1
        ) {
            print_progress_bar(
                contador + 1,
                TOTAL_FRAMES,
                elapsed,
                filename
            );
        }

        int key =
            cv::waitKey(1) & 0xFF;

        if (key == 27 || key == 'q') {
            break;
        }
    }

    std::cout << std::endl;

    out.release();

    cudaFree(d_frame);
    cudaFree(d_depth);
    cudaFree(d_faces);

    cv::destroyAllWindows();

    std::cout
        << "Video guardado en: "
        << filename
        << std::endl;

    return 0;
}
