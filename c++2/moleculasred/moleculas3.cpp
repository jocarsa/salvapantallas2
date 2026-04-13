#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <omp.h>

using namespace cv;
using namespace std;

// Simulation parameters
const int WIDTH       = 3840;
const int HEIGHT      = 2160;
const int NUM_CIRCLES = 10000;
const int FRAME_RATE  = 60;
const int NUM_FRAMES  = FRAME_RATE * 60;  // 1 minute
const double FADE     = 0.1;              // 10% fade

// Spatial-hash grid
const int CELL_SIZE = 30;
const int GRID_W    = (WIDTH  + CELL_SIZE - 1) / CELL_SIZE;
const int GRID_H    = (HEIGHT + CELL_SIZE - 1) / CELL_SIZE;

// Bounding the frame queue size to limit RAM
const size_t MAX_QUEUE_SIZE = 10;

struct Circle {
    float x, y, dir;
    Circle(int w, int h) {
        x = rand() / float(RAND_MAX) * w;
        y = rand() / float(RAND_MAX) * h;
        dir = rand() / float(RAND_MAX) * 2*CV_PI;
    }
};

struct Line { Point a, b; Line(const Point& p1, const Point& p2):a(p1),b(p2){} };

int main(){
    srand(unsigned(time(nullptr)));

    // Per-thread RNGs
    int nThreads = omp_get_max_threads();
    vector<mt19937> rngs(nThreads);
    unsigned seed0 = unsigned(time(nullptr));
    for(int t=0;t<nThreads;t++) rngs[t].seed(seed0 ^ (t*0x9e3779b1));

    // Video writer
    string filename = "video_" + to_string(time(nullptr)) + ".mp4";
    VideoWriter writer(
        filename,
        VideoWriter::fourcc('m','p','4','v'),
        FRAME_RATE,
        Size(WIDTH,HEIGHT)
    );
    if(!writer.isOpened()){
        cerr<<"Cannot open video for writing"<<endl;
        return -1;
    }

    // Thread-safe frame queue
    queue<Mat> frameQ;
    mutex mtx;
    condition_variable cv_push, cv_pop;
    bool done = false;

    // Encoder thread
    thread encoder([&]{
        while(true){
            Mat frame;
            {
                unique_lock<mutex> lk(mtx);
                cv_pop.wait(lk, [&]{ return !frameQ.empty() || done; });
                if(frameQ.empty() && done) break;
                frame = move(frameQ.front());
                frameQ.pop();
                cv_push.notify_one();
            }
            writer.write(frame);
        }
    });

    // Display window
    namedWindow("Framebuffer", WINDOW_AUTOSIZE);

    // Init canvas and circles
    Mat canvas(HEIGHT,WIDTH,CV_8UC3,Scalar(255,255,255));
    vector<Circle> circles;
    circles.reserve(NUM_CIRCLES);
    for(int i=0;i<NUM_CIRCLES;i++) circles.emplace_back(WIDTH,HEIGHT);

    vector<vector<int>> grid(GRID_W*GRID_H);

    // Main loop
    for(int f=0; f<NUM_FRAMES; f++){
        // Fade
        Mat white(HEIGHT,WIDTH,CV_8UC3,Scalar(255,255,255));
        addWeighted(canvas,1-FADE,white,FADE,0,canvas);

        // Move circles
        #pragma omp parallel for schedule(static)
        for(int i=0;i<NUM_CIRCLES;i++){
            int tid = omp_get_thread_num();
            float d = uniform_real_distribution<float>(-0.05f,0.05f)(rngs[tid]);
            circles[i].dir += d;
            circles[i].x += cos(circles[i].dir);
            circles[i].y += sin(circles[i].dir);
            if(circles[i].x<0||circles[i].x>WIDTH||circles[i].y<0||circles[i].y>HEIGHT)
                circles[i].dir += CV_PI;
        }

        // Build grid
        for(auto&cell:grid) cell.clear();
        for(int i=0;i<NUM_CIRCLES;i++){
            int gx=clamp(int(circles[i].x/CELL_SIZE),0,GRID_W-1);
            int gy=clamp(int(circles[i].y/CELL_SIZE),0,GRID_H-1);
            grid[gy*GRID_W+gx].push_back(i);
        }

        // Collision & record
        vector<Line> lines;
        vector<Point> pts;
        vector<pair<int,int>> bounces;
        pts.reserve(NUM_CIRCLES);
        lines.reserve(5000); bounces.reserve(500);

        #pragma omp parallel
        {
            int tid=omp_get_thread_num();
            vector<Line>  locL;
            vector<Point> locP;
            vector<pair<int,int>> locB;
            locP.reserve(NUM_CIRCLES/nThreads+1);
            locL.reserve(2000); locB.reserve(200);

            #pragma omp for schedule(static)
            for(int i=0;i<NUM_CIRCLES;i++){
                locP.emplace_back(cvRound(circles[i].x),cvRound(circles[i].y));
                int gx=clamp(int(circles[i].x/CELL_SIZE),0,GRID_W-1);
                int gy=clamp(int(circles[i].y/CELL_SIZE),0,GRID_H-1);
                for(int dy=-1;dy<=1;dy++){
                    int ny=gy+dy; if(ny<0||ny>=GRID_H) continue;
                    for(int dx=-1;dx<=1;dx++){
                        int nx=gx+dx; if(nx<0||nx>=GRID_W) continue;
                        for(int j: grid[ny*GRID_W+nx]){
                            if(j<=i) continue;
                            float dx_=circles[i].x-circles[j].x;
                            float dy_=circles[i].y-circles[j].y;
                            if(fabs(dx_)<109&&fabs(dy_)<109){
                                float dist=sqrt(dx_*dx_+dy_*dy_);
                                if(dist<10) locB.emplace_back(i,j);
                                else if(dist<50)
                                    locL.emplace_back(
                                        Point(cvRound(circles[i].x),cvRound(circles[i].y)),
                                        Point(cvRound(circles[j].x),cvRound(circles[j].y))
                                    );
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            {
                lines.insert(lines.end(),locL.begin(),locL.end());
                pts.insert(pts.end(),locP.begin(),locP.end());
                bounces.insert(bounces.end(),locB.begin(),locB.end());
            }
        }

        // Apply bounces
        for(auto&pr:bounces){
            int i=pr.first,j=pr.second;
            circles[i].dir+=CV_PI; circles[j].dir+=CV_PI;
            circles[i].x+=cos(circles[i].dir)*2;
            circles[i].y+=sin(circles[i].dir)*2;
            circles[j].x+=cos(circles[j].dir)*2;
            circles[j].y+=sin(circles[j].dir)*2;
        }

        // Draw
        for(auto&L:lines)
            line(canvas,L.a,L.b,Scalar(0,0,0),1,LINE_AA);
        for(auto&P:pts)
            circle(canvas,P,5,Scalar(0,0,0),-1,LINE_AA);

        // Enqueue frame, blocking if queue too large
        {
            unique_lock<mutex> lk(mtx);
            cv_push.wait(lk, [&]{ return frameQ.size() < MAX_QUEUE_SIZE; });
            frameQ.push(canvas.clone());
        }
        cv_pop.notify_one();

        // Show occasionally
        if(f%100==0){ imshow("Framebuffer",canvas); if(waitKey(1)==27) break; }
    }

    // Finish
    {
        unique_lock<mutex> lk(mtx);
        done = true;
    }
    cv_pop.notify_one();
    encoder.join();

    writer.release(); destroyAllWindows();
    cout<<"Saved "<<filename<<"\n";
}

