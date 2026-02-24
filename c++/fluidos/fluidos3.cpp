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

// Global Simulation Parameters
const float RATIO = 9.0f / 16.0f;
const int GRID_SIZE_X = 550;
const int GRID_SIZE_Y = static_cast<int>(std::round(GRID_SIZE_X * RATIO));
const int ITERATIONS = 16;
const float DIFFUSION = 0.00005f;
const float VISCOSITY = 0.00001f;
const float DENSITY_DECAY = 0.999f;
const float DT = 0.15f;

// Video parameters
const int VIDEO_WIDTH = 1920;
const int VIDEO_HEIGHT = 1080;
const int FPS = 60;
const int DURATION_MINUTES = 60;
const int TOTAL_FRAMES = FPS * 60 * DURATION_MINUTES;

// Helper: 2D-to-1D index conversion (grid stored in a vector)
inline int IX(int x, int y) {
    return y * (GRID_SIZE_X + 2) + x;
}

// Set boundary conditions (remains serial due to small loop sizes)
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

// Diffuse: Updates each cell based on neighbors (parallelized with OpenMP)
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

// Advect: Moves quantities along the velocity field (parallelized)
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

// Project: Enforces incompressibility using a Jacobi iteration (parallelized)
// Uses a temporary array (new_p) to allow safe parallel updates
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
    // Swap arrays (copy content)
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

// Density step: applies decay and then diffuses and advects the density field
void density_step(std::vector<float>& d, std::vector<float>& d_prev,
                  const std::vector<float>& u, const std::vector<float>& v) {
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

// Add fluid source: introduces swirling sources and occasional random bursts
void add_fluid_source(std::vector<float>& density,
                      std::vector<float>& vx,
                      std::vector<float>& vy,
                      std::vector<float>& hue,
                      int frame) {
    float t = frame / static_cast<float>(FPS);
    int num_swirls = 25;
    for (int swirl_id = 0; swirl_id < num_swirls; ++swirl_id) {
        float angle_offset = (swirl_id / static_cast<float>(num_swirls)) * 2 * static_cast<float>(M_PI);
        int cx = GRID_SIZE_X / 2 + static_cast<int>(std::cos(t * 0.2f + angle_offset) * GRID_SIZE_X * 0.3f);
        int cy = GRID_SIZE_Y / 2 + static_cast<int>(std::sin(t * 0.3f + angle_offset) * GRID_SIZE_Y * 0.3f);
        cx = std::max(1, std::min(GRID_SIZE_X, cx));
        cy = std::max(1, std::min(GRID_SIZE_Y, cy));
        int idx = IX(cx, cy);
        density[idx] += 200.0f;
        float angle = t * 2.0f + swirl_id * 0.5f;
        float force = 50.0f + 20.0f * std::sin(t * 0.5f);
        vx[idx] = std::cos(angle) * force;
        vy[idx] = std::sin(angle) * force;
        hue[idx] = std::fmod(swirl_id * 60.0f + t * 10.0f, 360.0f);
    }
    // Occasionally add random bursts
    if (static_cast<float>(std::rand()) / RAND_MAX < 0.03f) {
        int rand_i = std::rand() % GRID_SIZE_X + 1;
        int rand_j = std::rand() % GRID_SIZE_Y + 1;
        int idx = IX(rand_i, rand_j);
        density[idx] += 500.0f;
        float angle = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * static_cast<float>(M_PI);
        float force = 50.0f + (static_cast<float>(std::rand()) / RAND_MAX) * 100.0f;
        vx[idx] += std::cos(angle) * force;
        vy[idx] += std::sin(angle) * force;
        hue[idx] = (static_cast<float>(std::rand()) / RAND_MAX) * 360.0f;
    }
}

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

// Render: Creates an image from the simulation state (parallelized loop)
cv::Mat render(const std::vector<float>& density, const std::vector<float>& hue,
               float cell_size_x, float cell_size_y) {
    cv::Mat img(VIDEO_HEIGHT, VIDEO_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int j = 1; j <= GRID_SIZE_Y; ++j) {
        for (int i = 1; i <= GRID_SIZE_X; ++i) {
            int idx = IX(i, j);
            float d = density[idx];
            if (d > 0.1f) {
                float alpha = std::min(d / 200.0f, 1.0f);
                float h_val = hue[idx];
                float s = std::min(100.0f, 70.0f + d / 10.0f);
                float l = std::min(70.0f, 40.0f + d / 20.0f);
                cv::Vec3b color = hsl_to_rgb(h_val, s, l);
                
                // Compute the exact cell boundaries using floor and ceil
                int x0 = static_cast<int>(std::floor((i - 1) * cell_size_x));
                int y0 = static_cast<int>(std::floor((j - 1) * cell_size_y));
                int x1 = static_cast<int>(std::ceil(i * cell_size_x));
                int y1 = static_cast<int>(std::ceil(j * cell_size_y));
                // Ensure that the ROI stays within the image boundaries
                x1 = std::min(x1, VIDEO_WIDTH);
                y1 = std::min(y1, VIDEO_HEIGHT);
                
                cv::Rect roi_rect(x0, y0, x1 - x0, y1 - y0);
                cv::Mat roi = img(roi_rect);
                cv::Mat color_rect(roi.size(), roi.type(), cv::Scalar(color[0], color[1], color[2]));
                cv::addWeighted(roi, 1 - alpha, color_rect, alpha, 0, roi);
            }
        }
    }
    return img;
}


// Main simulation loop and video output
int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    
    int size = (GRID_SIZE_X + 2) * (GRID_SIZE_Y + 2);
    std::vector<float> density(size, 0.0f);
    std::vector<float> density_prev(size, 0.0f);
    std::vector<float> velocity_x(size, 0.0f);
    std::vector<float> velocity_y(size, 0.0f);
    std::vector<float> velocity_prev_x(size, 0.0f);
    std::vector<float> velocity_prev_y(size, 0.0f);
    std::vector<float> hue(size, 0.0f);
    std::vector<float> hue_prev(size, 0.0f);
    
    // Compute cell sizes for full-screen rendering
    float cell_size_x = static_cast<float>(VIDEO_WIDTH) / GRID_SIZE_X;
    float cell_size_y = static_cast<float>(VIDEO_HEIGHT) / GRID_SIZE_Y;
    
    std::time_t now = std::time(nullptr);  // Get the current epoch time
    std::string output_filename = "fluid_simulation_" + std::to_string(now) + ".mp4";
    cv::VideoWriter videoWriter(output_filename,
                                cv::VideoWriter::fourcc('m','p','4','v'),
                                FPS, cv::Size(VIDEO_WIDTH, VIDEO_HEIGHT));
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open video writer" << std::endl;
        return -1;
    }
    
    std::cout << "Starting simulation: " << TOTAL_FRAMES << " frames at " << FPS << " FPS" << std::endl;
    std::cout << "Output file: " << output_filename << std::endl;
    std::cout << "Grid size: " << GRID_SIZE_X << "x" << GRID_SIZE_Y << " cells" << std::endl;
    std::cout << "Cell size: " << cell_size_x << "x" << cell_size_y << " pixels" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    
    for (int frame = 0; frame < TOTAL_FRAMES; ++frame) {
        add_fluid_source(density, velocity_x, velocity_y, hue, frame);
        
        velocity_step(velocity_x, velocity_y, velocity_prev_x, velocity_prev_y);
        density_step(density, density_prev, velocity_x, velocity_y);
        hue_step(hue, hue_prev, velocity_x, velocity_y);
        
        cv::Mat img = render(density, hue, cell_size_x, cell_size_y);
        
        int frames_remaining = TOTAL_FRAMES - frame - 1;
        int minutes_remaining = frames_remaining / (FPS * 60);
        int seconds_remaining = (frames_remaining % (FPS * 60)) / FPS;
        std::string progress_text = "Frame: " + std::to_string(frame + 1) + "/" + std::to_string(TOTAL_FRAMES)
                                    + " | Time remaining: " + std::to_string(minutes_remaining) + "m "
                                    + std::to_string(seconds_remaining) + "s";
        /*cv::putText(img, progress_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(255, 255, 255), 2);*/
        
        videoWriter.write(img);
        cv::imshow("Fluid Simulation (Fullscreen)", img);
        if (cv::waitKey(1) == 'q')
            break;
        
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() >= 5.0) {
            double fps_actual = frame_count / elapsed.count();
            std::cout << "Current FPS: " << fps_actual << " | Progress: " << frame+1 << "/" << TOTAL_FRAMES
                      << " frames (" << ((frame+1)*100.0/TOTAL_FRAMES) << "%)" << std::endl;
            frame_count = 0;
            start_time = current_time;
        }
    }
    
    videoWriter.release();
    cv::destroyAllWindows();
    std::cout << "Simulation complete. Video saved to " << output_filename << std::endl;
    return 0;
}

