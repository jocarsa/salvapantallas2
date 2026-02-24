#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <string>
#include <omp.h>

// New simulation parameters
const int CELL_SIZE = 8; // each simulation cell is CELL_SIZE x CELL_SIZE pixels
const int VIDEO_WIDTH = 1920;
const int VIDEO_HEIGHT = 1080;
const int GRID_SIZE_X = VIDEO_WIDTH / CELL_SIZE;
const int GRID_SIZE_Y = VIDEO_HEIGHT / CELL_SIZE;

const int ITERATIONS = 16;
const float DIFFUSION = 0.00005f;
const float VISCOSITY = 0.00001f;
// Set this to 0.9 to see the decay issue—now fixed via improved rendering.
const float DENSITY_DECAY = 0.998f;  
const float DT = 0.15f;

// New parameter: number of emitters (can be modified)
const int EMITTER_COUNT = 25;

// Video output parameters
const int FPS = 60;
const int DURATION_MINUTES = 60;
const int TOTAL_FRAMES = FPS * 60 * DURATION_MINUTES;

// Helper: 2D-to-1D index conversion (grid stored in a vector)
inline int IX(int x, int y) {
    return y * (GRID_SIZE_X + 2) + x;
}

// Set boundary conditions
void set_boundary(int b, std::vector<float>& x) {
    // Vertical edges
    for (int j = 1; j <= GRID_SIZE_Y; ++j) {
        x[IX(0, j)] = (b == 1) ? -x[IX(1, j)] : x[IX(1, j)];
        x[IX(GRID_SIZE_X + 1, j)] = (b == 1) ? -x[IX(GRID_SIZE_X, j)] : x[IX(GRID_SIZE_X, j)];
    }
    // Horizontal edges
    for (int i = 1; i <= GRID_SIZE_X; ++i) {
        x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, GRID_SIZE_Y + 1)] = (b == 2) ? -x[IX(i, GRID_SIZE_Y)] : x[IX(i, GRID_SIZE_Y)];
    }
    // Corners
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, GRID_SIZE_Y + 1)] = 0.5f * (x[IX(1, GRID_SIZE_Y + 1)] + x[IX(0, GRID_SIZE_Y)]);
    x[IX(GRID_SIZE_X + 1, 0)] = 0.5f * (x[IX(GRID_SIZE_X, 0)] + x[IX(GRID_SIZE_X + 1, 1)]);
    x[IX(GRID_SIZE_X + 1, GRID_SIZE_Y + 1)] = 0.5f * (x[IX(GRID_SIZE_X, GRID_SIZE_Y + 1)] + x[IX(GRID_SIZE_X + 1, GRID_SIZE_Y)]);
}

// Diffuse: updates each cell based on neighbors (parallelized with OpenMP)
void diffuse(int b, std::vector<float>& x, const std::vector<float>& x0, float diff) {
    float a = diff * GRID_SIZE_X * GRID_SIZE_Y / (((GRID_SIZE_X + GRID_SIZE_Y) / 2.0f));
    for (int k = 0; k < ITERATIONS; ++k) {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j <= GRID_SIZE_Y; ++j) {
            for (int i = 1; i <= GRID_SIZE_X; ++i) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (
                    x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                    x[IX(i, j - 1)] + x[IX(i, j + 1)]
                )) / (1 + 4 * a);
            }
        }
        set_boundary(b, x);
    }
}

// Advect: moves quantities along the velocity field (parallelized)
void advect(int b, std::vector<float>& d, const std::vector<float>& d0,
            const std::vector<float>& u, const std::vector<float>& v) {
    #pragma omp parallel for collapse(2)
    for (int j = 1; j <= GRID_SIZE_Y; ++j) {
        for (int i = 1; i <= GRID_SIZE_X; ++i) {
            float x_pos = i - DT * u[IX(i, j)];
            float y_pos = j - DT * v[IX(i, j)];
            // Clamp coordinates to [0.5, GRID_SIZE + 0.5]
            x_pos = std::max(0.5f, std::min(x_pos, static_cast<float>(GRID_SIZE_X) + 0.5f));
            y_pos = std::max(0.5f, std::min(y_pos, static_cast<float>(GRID_SIZE_Y) + 0.5f));
            
            int i0 = static_cast<int>(x_pos);
            int i1 = i0 + 1;
            int j0 = static_cast<int>(y_pos);
            int j1 = j0 + 1;
            
            float s1 = x_pos - i0;
            float s0 = 1 - s1;
            float t1 = y_pos - j0;
            float t0 = 1 - t1;
            
            d[IX(i, j)] =
                s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_boundary(b, d);
}

// Project: enforces incompressibility (parallelized Jacobi iteration)
void project(std::vector<float>& u, std::vector<float>& v,
             std::vector<float>& p, std::vector<float>& div) {
    float h_x = 1.0f / GRID_SIZE_X;
    float h_y = 1.0f / GRID_SIZE_Y;
    
    #pragma omp parallel for collapse(2)
    for (int j = 1; j <= GRID_SIZE_Y; ++j) {
        for (int i = 1; i <= GRID_SIZE_X; ++i) {
            div[IX(i, j)] = -0.5f * (
                h_x * (u[IX(i + 1, j)] - u[IX(i - 1, j)]) +
                h_y * (v[IX(i, j + 1)] - v[IX(i, j - 1)])
            );
            p[IX(i, j)] = 0.0f;
        }
    }
    set_boundary(0, div);
    set_boundary(0, p);
    
    std::vector<float> new_p(p.size(), 0.0f);
    for (int k = 0; k < ITERATIONS; ++k) {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j <= GRID_SIZE_Y; ++j) {
            for (int i = 1; i <= GRID_SIZE_X; ++i) {
                new_p[IX(i, j)] = (div[IX(i, j)] +
                                   p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                                   p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4.0f;
            }
        }
        p = new_p;
        set_boundary(0, p);
    }
    
    #pragma omp parallel for collapse(2)
    for (int j = 1; j <= GRID_SIZE_Y; ++j) {
        for (int i = 1; i <= GRID_SIZE_X; ++i) {
            u[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h_x;
            v[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h_y;
        }
    }
    set_boundary(1, u);
    set_boundary(2, v);
}

// Velocity step: updates the velocity field
void velocity_step(std::vector<float>& u, std::vector<float>& v,
                   std::vector<float>& u_prev, std::vector<float>& v_prev) {
    std::vector<float> temp = u;
    u = u_prev;
    u_prev = temp;
    temp = v;
    v = v_prev;
    v_prev = temp;
    
    diffuse(1, u, u_prev, VISCOSITY);
    diffuse(2, v, v_prev, VISCOSITY);
    
    std::vector<float> p(u.size(), 0.0f), div(u.size(), 0.0f);
    project(u, v, p, div);
    
    temp = u;
    u = u_prev;
    u_prev = temp;
    temp = v;
    v = v_prev;
    v_prev = temp;
    
    advect(1, u, u_prev, u_prev, v_prev);
    advect(2, v, v_prev, u_prev, v_prev);
    
    project(u, v, p, div);
}

// Density step: applies decay then diffuses and advects the density field
void density_step(std::vector<float>& d, std::vector<float>& d_prev,
                  const std::vector<float>& u, const std::vector<float>& v) {
    // Apply multiplicative decay
    #pragma omp parallel for
    for (size_t i = 0; i < d.size(); ++i)
        d[i] *= DENSITY_DECAY;
    
    d_prev = d;  // copy
    diffuse(0, d, d_prev, DIFFUSION);
    
    d_prev = d;  // copy again
    advect(0, d, d_prev, u, v);
}

// Hue step: similar to density step (for coloring the fluid)
void hue_step(std::vector<float>& hue, std::vector<float>& hue_prev,
              const std::vector<float>& u, const std::vector<float>& v) {
    hue_prev = hue;  // copy
    diffuse(0, hue, hue_prev, DIFFUSION * 0.1f);
    
    hue_prev = hue;  // copy again
    advect(0, hue, hue_prev, u, v);
}

// ----------------------------------------------------------------
// Emitter Structure & Functions for Independent Emission
// ----------------------------------------------------------------

// Define an Emitter with independent properties.
struct Emitter {
    float x, y;       // Current position on the simulation grid ([1, GRID_SIZE_X], [1, GRID_SIZE_Y])
    float vx, vy;     // Velocity for independent random movement
    float hue;        // Hue value for color emission
    float force;      // Force magnitude for velocity injection

    // Constructor with randomized initialization (adjust ranges as needed)
    Emitter() {
        x = 1 + std::rand() % GRID_SIZE_X;
        y = 1 + std::rand() % GRID_SIZE_Y;
        // Small random velocity for movement:
        vx = (std::rand() % 100 - 50) / 500.0f;
        vy = (std::rand() % 100 - 50) / 500.0f;
        hue = std::rand() % 360;        // Hue between 0 and 359
        force = 50.0f + (std::rand() % 50);  // Force between 50 and 100
    }
};

// Global vector of emitters; will be initialized in main().
std::vector<Emitter> emitters;

// Update emitter positions and properties every frame.
void update_emitters() {
    for (auto &e : emitters) {
        // Update position according to velocity.
        e.x += e.vx;
        e.y += e.vy;
        
        // Bounce off the boundaries to stay within [1, GRID_SIZE]
        if (e.x < 1) { e.x = 1; e.vx = -e.vx; }
        if (e.x > GRID_SIZE_X) { e.x = GRID_SIZE_X; e.vx = -e.vx; }
        if (e.y < 1) { e.y = 1; e.vy = -e.vy; }
        if (e.y > GRID_SIZE_Y) { e.y = GRID_SIZE_Y; e.vy = -e.vy; }
        
        // Optionally, slowly shift the hue over time.
        e.hue = std::fmod(e.hue + 0.5f, 360.0f);
    }
}

// Modified add_fluid_source using per-emitter properties.
void add_fluid_source(std::vector<float>& density,
                      std::vector<float>& vx,
                      std::vector<float>& vy,
                      std::vector<float>& hue,
                      int frame) {
    // First, update the emitters.
    update_emitters();

    // For each emitter, inject density, velocity, and hue into the simulation.
    for (const auto &e : emitters) {
        // Convert emitter's floating-point position to grid cell indices.
        int cx = static_cast<int>(e.x);
        int cy = static_cast<int>(e.y);
        // Clamp indices to [1, GRID_SIZE]
        cx = std::max(1, std::min(GRID_SIZE_X, cx));
        cy = std::max(1, std::min(GRID_SIZE_Y, cy));
        int idx = IX(cx, cy);
        
        // Inject density (adjust the injection amount as desired).
        density[idx] += 200.0f;
        
        // Inject velocity based on the emitter's force and direction (using its hue as angle).
        float angle_rad = e.hue * static_cast<float>(M_PI) / 180.0f;
        vx[idx] += std::cos(angle_rad) * e.force;
        vy[idx] += std::sin(angle_rad) * e.force;
        
        // Set the hue for the cell to the emitter's hue.
        hue[idx] = e.hue;
    }
    
    // Occasionally add a random burst (optional)
    if (static_cast<float>(std::rand()) / RAND_MAX < 0.03f) {
        int rand_i = std::rand() % GRID_SIZE_X + 1;
        int rand_j = std::rand() % GRID_SIZE_Y + 1;
        int idx = IX(rand_i, rand_j);
        density[idx] += 500.0f;
        float angle = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * static_cast<float>(M_PI);
        vx[idx] += std::cos(angle) * 100.0f;
        vy[idx] += std::sin(angle) * 100.0f;
        hue[idx] = (static_cast<float>(std::rand()) / RAND_MAX) * 360.0f;
    }
}

// ----------------------------------------------------------------
// Utility Functions for Rendering
// ----------------------------------------------------------------

// HSL to RGB conversion (returns BGR as OpenCV expects)
cv::Vec3b hsl_to_rgb(float h, float s, float l) {
    h = std::fmod(h, 360.0f);
    s = std::min(100.0f, std::max(0.0f, s)) / 100.0f;
    l = std::min(100.0f, std::max(0.0f, l)) / 100.0f;
    
    float c = (1 - std::fabs(2 * l - 1)) * s;
    float x = c * (1 - std::fabs(std::fmod(h / 60.0f, 2) - 1));
    float m = l - c / 2;
    
    float r, g, b;
    if (h < 60)
        r = c, g = x, b = 0;
    else if (h < 120)
        r = x, g = c, b = 0;
    else if (h < 180)
        r = 0, g = c, b = x;
    else if (h < 240)
        r = 0, g = x, b = c;
    else if (h < 300)
        r = x, g = 0, b = c;
    else
        r = c, g = 0, b = x;
    
    int R = std::round((r + m) * 255);
    int G = std::round((g + m) * 255);
    int B = std::round((b + m) * 255);
    
    R = std::min(255, std::max(0, R));
    G = std::min(255, std::max(0, G));
    B = std::min(255, std::max(0, B));
    
    return cv::Vec3b(B, G, R);
}

// Render: creates an image from the simulation state.
// Here we modify the rendering so that even low density values produce a fading color,
// rather than abruptly turning to black.
cv::Mat render(const std::vector<float>& density, const std::vector<float>& hue) {
    cv::Mat img(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int j = 1; j <= GRID_SIZE_Y; ++j) {
        for (int i = 1; i <= GRID_SIZE_X; ++i) {
            int idx = IX(i, j);
            float d = density[idx];
            // If density is extremely low, skip drawing.
            if (d < 0.001f) continue;
            float h_val = hue[idx];
            float s = std::min(100.0f, 70.0f + d / 10.0f);
            float l = std::min(70.0f, 40.0f + d / 20.0f);
            
            // Get base color from HSL
            cv::Vec3b base_color = hsl_to_rgb(h_val, s, l);
            // Compute brightness factor. Adjust the divisor (here 200.0f) if you want a different fade curve.
            float brightness = std::min(1.0f, d / 200.0f);
            cv::Vec3b color(static_cast<uchar>(base_color[0] * brightness),
                            static_cast<uchar>(base_color[1] * brightness),
                            static_cast<uchar>(base_color[2] * brightness));
            
            int x0 = (i - 1) * CELL_SIZE;
            int y0 = (j - 1) * CELL_SIZE;
            int x1 = i * CELL_SIZE;
            int y1 = j * CELL_SIZE;
            // Clamp to image boundaries
            x1 = std::min(x1, VIDEO_WIDTH);
            y1 = std::min(y1, VIDEO_HEIGHT);
            
            cv::Rect roi_rect(x0, y0, x1 - x0, y1 - y0);
            cv::Mat roi = img(roi_rect);
            roi.setTo(cv::Scalar(color[0], color[1], color[2]));
        }
    }
    return img;
}

// Helper function: formats seconds into HH:MM:SS string.
std::string format_time(double total_seconds) {
    int seconds = static_cast<int>(total_seconds);
    int hours = seconds / 3600;
    int minutes = (seconds % 3600) / 60;
    int secs = seconds % 60;
    char buffer[9]; // HH:MM:SS
    std::snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", hours, minutes, secs);
    return std::string(buffer);
}

// ----------------------------------------------------------------
// Main Simulation
// ----------------------------------------------------------------
int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    // Initialize emitter vector based on the EMITTER_COUNT parameter.
    emitters.reserve(EMITTER_COUNT);
    for (int i = 0; i < EMITTER_COUNT; ++i) {
        emitters.push_back(Emitter());
    }
    
    int size = (GRID_SIZE_X + 2) * (GRID_SIZE_Y + 2);
    std::vector<float> density(size, 0.0f);
    std::vector<float> density_prev(size, 0.0f);
    std::vector<float> velocity_x(size, 0.0f);
    std::vector<float> velocity_y(size, 0.0f);
    std::vector<float> velocity_prev_x(size, 0.0f);
    std::vector<float> velocity_prev_y(size, 0.0f);
    std::vector<float> hue(size, 0.0f);
    std::vector<float> hue_prev(size, 0.0f);
    
    std::time_t now = std::time(nullptr);
    std::string output_filename = "fluid_simulation_" + std::to_string(now) + ".mp4";
    cv::VideoWriter videoWriter(output_filename,
                                cv::VideoWriter::fourcc('m','p','4','v'),
                                FPS, cv::Size(VIDEO_WIDTH, VIDEO_HEIGHT));
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open video writer" << std::endl;
        return -1;
    }
    
    std::cout << "Starting simulation (" << TOTAL_FRAMES << " frames at " << FPS << " FPS)" << std::endl;
    std::cout << "Output file: " << output_filename << std::endl;
    std::cout << "Grid size: " << GRID_SIZE_X << " x " << GRID_SIZE_Y << " cells (" 
              << CELL_SIZE << "x" << CELL_SIZE << " pixels per cell)" << std::endl;
    std::cout << "Number of Emitters: " << EMITTER_COUNT << std::endl;
    
    auto simulation_start = std::chrono::high_resolution_clock::now();
    auto progress_last = simulation_start;
    int frame = 0;
    
    for (; frame < TOTAL_FRAMES; ++frame) {
        add_fluid_source(density, velocity_x, velocity_y, hue, frame);
        velocity_step(velocity_x, velocity_y, velocity_prev_x, velocity_prev_y);
        density_step(density, density_prev, velocity_x, velocity_y);
        hue_step(hue, hue_prev, velocity_x, velocity_y);
        
        cv::Mat img = render(density, hue);
        videoWriter.write(img);
        
        cv::imshow("Fluid Simulation", img);
        if (cv::waitKey(1) == 'q')
            break;
        
        // Every 5 seconds update console progress.
        auto now_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> interval = now_time - progress_last;
        if (interval.count() >= 5.0) {
            std::chrono::duration<double> sim_elapsed = now_time - simulation_start;
            double avg_frame_time = sim_elapsed.count() / (frame + 1);
            double est_remaining = avg_frame_time * (TOTAL_FRAMES - (frame + 1));
            double progress_pct = 100.0 * (frame + 1) / TOTAL_FRAMES;
            std::cout << "Progress: " << progress_pct << "% | "
                      << "Elapsed: " << format_time(sim_elapsed.count()) << " | "
                      << "Remaining: " << format_time(est_remaining) << " | "
                      << "FPS: " << (1.0 / avg_frame_time) << std::endl;
            progress_last = now_time;
        }
    }
    
    videoWriter.release();
    cv::destroyAllWindows();
    std::cout << "Simulation complete. Video saved to " << output_filename << std::endl;
    return 0;
}

