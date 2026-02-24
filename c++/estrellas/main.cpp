#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>

class Estrella {
public:
    double angulo;
    double distancia;
    double velocidad;
    double radius; // Current radius of the star

    Estrella() {
        angulo    = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
        distancia = static_cast<double>(rand()) / RAND_MAX * 1000.0;
        velocidad = 0.001 + static_cast<double>(rand()) / RAND_MAX * 0.005;
        radius    = 1.0; // Start with the minimum radius
    }
};

void applyBoxBlur(const cv::Mat& src, cv::Mat& dst, int kernelSize) {
    cv::blur(src, dst, cv::Size(kernelSize, kernelSize));
}

int main() {
    // Seed the C library RNG for Estrella constructors
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Use mt19937 to pick a random number of stars
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10000, 50000);
    int numero_estrellas = dis(gen);
    std::vector<Estrella> estrellas(numero_estrellas);

    const int anchura   = 1920;
    const int altura    = 1080;
    const double maxDistance = 1000.0;
    const double minRadius   = 0.1;
    const double maxRadius   = 2.0;
    const double smoothing   = 0.02;    // Slower interpolation
    const int    SUBPIXEL_SHIFT = 4;    // 4 bits for sub-pixel precision

    // Prepare video writer
    std::time_t epoch_time = std::time(nullptr);
    std::string nombre_temp = "estrellas_temp_" + std::to_string(epoch_time) + ".mp4";
    cv::VideoWriter video(nombre_temp,
                          cv::VideoWriter::fourcc('m','p','4','v'),
                          60,
                          cv::Size(anchura, altura));

    // Frame buffers for motion blur + glow
    cv::Mat motionBlurBuf = cv::Mat::zeros(altura, anchura, CV_8UC3);

    try {
        // Render 1 minute at 60 FPS
        const int totalFrames = 60 * 60*60;
        for (int frameIdx = 0; frameIdx < totalFrames; ++frameIdx) {
            cv::Mat frame = cv::Mat::zeros(altura, anchura, CV_8UC3);

            // Update and draw each star
            for (auto& estrella : estrellas) {
                // Exponential motion
                estrella.distancia += estrella.velocidad * estrella.distancia;
                if (estrella.distancia > maxDistance)
                    estrella.distancia = 0.1;

                // Compute target radius based on distance
                double targetRadius = minRadius + (maxRadius - minRadius) *
                                      (estrella.distancia / maxDistance);
                // Smoothly move toward target
                estrella.radius += smoothing * (targetRadius - estrella.radius);

                // Compute sub-pixel accurate center & radius
                int scale    = 1 << SUBPIXEL_SHIFT;
                cv::Point centerFixed(
                    static_cast<int>((anchura*0.5 + estrella.distancia * std::cos(estrella.angulo)) * scale + 0.5),
                    static_cast<int>((altura*0.5 + estrella.distancia * std::sin(estrella.angulo)) * scale + 0.5)
                );
                int fixedRadius = static_cast<int>(estrella.radius * scale + 0.5);

                // Draw anti-aliased circle with sub-pixel precision
                cv::circle(frame,
                           centerFixed,
                           fixedRadius,
                           cv::Scalar(255, 255, 255),
                           -1,
                           cv::LINE_AA,
                           SUBPIXEL_SHIFT);
            }

            // Motion blur blending
            cv::addWeighted(motionBlurBuf, 0.7, frame, 0.3, 0.0, motionBlurBuf);
            // Glow effect via box blur
            cv::Mat glowBuf;
            applyBoxBlur(motionBlurBuf, glowBuf, 5);
            cv::addWeighted(motionBlurBuf, 1.0, glowBuf,    1.0, 0.0, glowBuf);

            // Write and display
            video.write(glowBuf);
            cv::imshow("Estrellas", glowBuf);
            if (cv::waitKey(1) == 'q')
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }

    video.release();
    cv::destroyAllWindows();

    // Rename to final location
    std::string nombre_final = "C:/render/estrellas_" + std::to_string(epoch_time) + ".mp4";
    if (std::rename(nombre_temp.c_str(), nombre_final.c_str()) == 0) {
        std::cout << "Video saved as '" << nombre_final << "'\n";
    } else {
        std::cerr << "Could not save the video." << std::endl;
    }

    return 0;
}

