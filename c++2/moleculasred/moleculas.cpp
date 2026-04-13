#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// Video dimensions and simulation parameters:
const int WIDTH = 1920;
const int HEIGHT = 1080;
const int NUM_CIRCLES = 1540;   // Number of circles to simulate.
const int FRAME_RATE = 60;      // Frames per second.
const int NUM_FRAMES = 60*60;     // Total frames (e.g., 600 frames = 10 seconds).
const double FADE = 0.1;        // Fade factor: 0.1 means 10% white overlay per frame.

struct Circle {
    float x, y;
    float direction;
    int r, g, b;
    float a;

    // Constructor: initialize with random position, direction, and color.
    Circle(int width, int height) {
        x = static_cast<float>(rand()) / RAND_MAX * width;
        y = static_cast<float>(rand()) / RAND_MAX * height;
        direction = static_cast<float>(rand()) / RAND_MAX * 2.0f * static_cast<float>(CV_PI);
        r = rand() % 256;
        g = rand() % 256;
        b = rand() % 256;
        a = 0.5f;
    }
};

int main() {
    // Seed random generator.
    srand(static_cast<unsigned int>(time(nullptr)));

    // Create output video filename with epoch timestamp.
    time_t epoch = time(nullptr);
    string filename = "video_" + to_string(epoch) + ".mp4";

    // Setup VideoWriter using mp4v codec.
    int codec = VideoWriter::fourcc('m','p','4','v');
    VideoWriter writer(filename, codec, FRAME_RATE, Size(WIDTH, HEIGHT));
    if (!writer.isOpened()){
        cerr << "Error: Could not open the output video file for writing" << endl;
        return -1;
    }

    // Create a window to display the framebuffer.
    namedWindow("Framebuffer", WINDOW_AUTOSIZE);

    // Create a persistent canvas initialized to white.
    Mat canvas(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));

    // Initialize circles.
    vector<Circle> circles;
    circles.reserve(NUM_CIRCLES);
    for (int i = 0; i < NUM_CIRCLES; i++){
        circles.push_back(Circle(WIDTH, HEIGHT));
    }

    // Main simulation loop.
    for (int frameIdx = 0; frameIdx < NUM_FRAMES; frameIdx++){
        // Fade previous drawings by overlaying a semi-transparent white rectangle.
        Mat whiteOverlay(canvas.size(), canvas.type(), Scalar(255,255,255));
        // The canvas is blended with white: 1-FADE (canvas) and FADE (whiteOverlay).
        addWeighted(canvas, 1.0 - FADE, whiteOverlay, FADE, 0, canvas);

        // Update each circle's position and direction.
        for (int i = 0; i < NUM_CIRCLES; i++){
            // Small random change in direction.
            float delta = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            circles[i].direction += delta;
            circles[i].x += cos(circles[i].direction);
            circles[i].y += sin(circles[i].direction);

            // Bounce off the boundaries.
            if (circles[i].x > WIDTH || circles[i].x < 0 ||
                circles[i].y > HEIGHT || circles[i].y < 0) {
                circles[i].direction += static_cast<float>(CV_PI);
            }
        }

        // Check for collisions and draw connecting lines.
        for (int i = 0; i < NUM_CIRCLES; i++){
            for (int j = i + 1; j < NUM_CIRCLES; j++){
                if (abs(circles[i].x - circles[j].x) < 109 &&
                    abs(circles[i].y - circles[j].y) < 109) {

                    float dx = circles[i].x - circles[j].x;
                    float dy = circles[i].y - circles[j].y;
                    float dist = sqrt(dx * dx + dy * dy);

                    // If circles are very close, simulate a bounce.
                    if (dist < 10) {
                        circles[i].direction += static_cast<float>(CV_PI);
                        circles[j].direction += static_cast<float>(CV_PI);
                        circles[i].x += cos(circles[i].direction) * 2;
                        circles[i].y += sin(circles[i].direction) * 2;
                    }

                    // If circles are moderately close, draw a connecting line.
                    if (dist < 50) {
                        line(canvas,
                             Point(cvRound(circles[i].x), cvRound(circles[i].y)),
                             Point(cvRound(circles[j].x), cvRound(circles[j].y)),
                             Scalar(0, 0, 0), // Black line.
                             1,
                             LINE_AA);
                    }
                }
            }
        }

        // Draw circles as filled black circles.
        for (int i = 0; i < NUM_CIRCLES; i++){
            circle(canvas,
                   Point(cvRound(circles[i].x), cvRound(circles[i].y)),
                   5,
                   Scalar(0, 0, 0), // Black color.
                   -1,
                   LINE_AA);
        }

        // Write the current canvas to the video.
        writer.write(canvas);

        // Display the framebuffer.
        imshow("Framebuffer", canvas);
        int key = waitKey(100 / FRAME_RATE);
        if (key == 27) { // ESC key.
            cout << "ESC key pressed, exiting simulation early." << endl;
            break;
        }
    }

    writer.release();
    destroyAllWindows();
    cout << "Video saved to " << filename << endl;
    return 0;
}

