#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>
#include <omp.h>              // <-- include OpenMP

using namespace cv;
using namespace std;

// Video dimensions and simulation parameters:
const int WIDTH       = 1920;
const int HEIGHT      = 1080;
const int NUM_CIRCLES = 1540;   // Number of circles to simulate.
const int FRAME_RATE  = 60;     // Frames per second.
const int NUM_FRAMES  = 60*60;  // Total frames (e.g., 3600 for 1 minute).
const double FADE     = 0.1;    // 10% white overlay per frame.

struct Circle {
    float x, y;
    float direction;
    int r,g,b;
    float a;
    Circle(int width, int height) {
        x = static_cast<float>(rand())/RAND_MAX * width;
        y = static_cast<float>(rand())/RAND_MAX * height;
        direction = static_cast<float>(rand())/RAND_MAX * 2.0f * CV_PI;
        r = rand()%256; g = rand()%256; b = rand()%256;
        a = 0.5f;
    }
};

int main() {
    srand((unsigned)time(nullptr));

    time_t epoch = time(nullptr);
    string filename = "video_" + to_string(epoch) + ".mp4";
    int codec = VideoWriter::fourcc('m','p','4','v');
    VideoWriter writer(filename, codec, FRAME_RATE, Size(WIDTH, HEIGHT));
    if (!writer.isOpened()){
        cerr << "Error: Could not open the output video file for writing" << endl;
        return -1;
    }

    namedWindow("Framebuffer", WINDOW_AUTOSIZE);
    Mat canvas(HEIGHT, WIDTH, CV_8UC3, Scalar(255,255,255));

    vector<Circle> circles;
    circles.reserve(NUM_CIRCLES);
    for (int i = 0; i < NUM_CIRCLES; i++)
        circles.emplace_back(WIDTH, HEIGHT);

    // Main simulation loop.
    for (int frameIdx = 0; frameIdx < NUM_FRAMES; frameIdx++) {
        // 1) Fade previous frame
        Mat whiteOverlay(canvas.size(), canvas.type(), Scalar(255,255,255));
        addWeighted(canvas, 1.0 - FADE, whiteOverlay, FADE, 0, canvas);

        // 2) Update positions in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_CIRCLES; i++) {
            float delta = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
            circles[i].direction += delta;
            circles[i].x += cos(circles[i].direction);
            circles[i].y += sin(circles[i].direction);
            // Bounce off walls
            if (circles[i].x < 0 || circles[i].x > WIDTH ||
                circles[i].y < 0 || circles[i].y > HEIGHT) {
                circles[i].direction += CV_PI;
            }
        }

        // 3) Collision detection + line‐drawing
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_CIRCLES; i++) {
            for (int j = i+1; j < NUM_CIRCLES; j++) {
                float dx = circles[i].x - circles[j].x;
                float dy = circles[i].y - circles[j].y;
                if (fabs(dx) < 109 && fabs(dy) < 109) {
                    float dist = sqrt(dx*dx + dy*dy);
                    if (dist < 10) {
                        // bounce both circles
                        #pragma omp critical
                        {
                            circles[i].direction += CV_PI;
                            circles[j].direction += CV_PI;
                            circles[i].x += cos(circles[i].direction)*2;
                            circles[i].y += sin(circles[i].direction)*2;
                        }
                    }
                    if (dist < 50) {
                        // draw connecting line
                        #pragma omp critical
                        {
                            line(canvas,
                                 Point(cvRound(circles[i].x), cvRound(circles[i].y)),
                                 Point(cvRound(circles[j].x), cvRound(circles[j].y)),
                                 Scalar(0,0,0),
                                 1, LINE_AA);
                        }
                    }
                }
            }
        }

        // 4) Draw circles in parallel (each draw is protected)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < NUM_CIRCLES; i++) {
            #pragma omp critical
            circle(canvas,
                   Point(cvRound(circles[i].x), cvRound(circles[i].y)),
                   5,
                   Scalar(0,0,0),
                   -1, LINE_AA);
        }

        // Write and display
        writer.write(canvas);
        imshow("Framebuffer", canvas);
        if (waitKey(100/FRAME_RATE) == 27) break;  // ESC
    }

    writer.release();
    destroyAllWindows();
    cout << "Video saved to " << filename << endl;
    return 0;
}

