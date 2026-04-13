
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <cctype>
#include <cerrno>
#include <cstring>
#ifdef __linux__
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
  std::exit(1); } } while(0)

__host__ __device__ inline float3 make_f3(float x, float y, float z) { return make_float3(x,y,z); }
__host__ __device__ inline float3 operator+(const float3& a, const float3& b){ return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__host__ __device__ inline float3 operator-(const float3& a, const float3& b){ return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__host__ __device__ inline float3 operator-(const float3& a){ return make_float3(-a.x,-a.y,-a.z); }
__host__ __device__ inline float3 operator*(const float3& a, float b){ return make_float3(a.x*b,a.y*b,a.z*b); }
__host__ __device__ inline float3 operator*(float b, const float3& a){ return make_float3(a.x*b,a.y*b,a.z*b); }
__host__ __device__ inline float3 operator*(const float3& a, const float3& b){ return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 operator/(const float3& a, float b){ return make_float3(a.x/b,a.y/b,a.z/b); }
__host__ __device__ inline float dot3(const float3& a, const float3& b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
__host__ __device__ inline float3 cross3(const float3& a, const float3& b){ return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__host__ __device__ inline float len3(const float3& a){ return sqrtf(dot3(a,a)); }
__host__ __device__ inline float3 norm3(const float3& a){ float l=len3(a); return (l>0)? a/l : make_float3(0,0,0); }
__host__ __device__ inline float3 min3(const float3& a, const float3& b){ return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)); }
__host__ __device__ inline float3 max3(const float3& a, const float3& b){ return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)); }
__host__ __device__ inline float clamp01(float x){ return fminf(1.0f, fmaxf(0.0f, x)); }
__host__ __device__ inline float3 clamp01(const float3& c){ return make_float3(clamp01(c.x), clamp01(c.y), clamp01(c.z)); }
__host__ __device__ inline float luminance(const float3& c){ return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

const int SCREEN_W = 1920;
const int SCREEN_H = 1080;
const int MAP_W_EXPECTED = 120;
const int MAP_H_EXPECTED = 120;
const int numAgents = 215;

const double SIM_TICK_SECONDS = 1.0;
const double SIM_TICKS_PER_REAL_SECOND = 6.0;
const int MAX_SIM_STEPS_PER_FRAME = 32;

const float TILE_WORLD = 1.0f;
const float FLOOR_THICKNESS_3D = 0.12f;
const float WALL_HEIGHT_3D = 0.65f;
const float WALL_HEIGHT_OFFSET_3D = 0.08f;
const float HUMAN_HEIGHT_3D = 1.45f;
const float HUMAN_RADIUS_3D = 0.16f;

const float CAMERA_NPC_ZOOM_FACTOR = 2.10f;
const float CAMERA_CENTER_SMOOTH = 0.045f;
const float CAMERA_ZOOM_SMOOTH   = 0.045f;
const float CAMERA_ANGLE_SMOOTH  = 0.085f;
const long long PLAYER_IDLE_FALLBACK_MS = 5000;
const float CAMERA_ZOOM_MIN = 0.12f;
const float CAMERA_ZOOM_MAX = 6.00f;
const float CAMERA_ZOOM_STEP = 0.18f;
const float CAMERA_ORBIT_STEP_DEG = 7.5f;
const float CAMERA_TILT_STEP = 0.08f;
const float PERSPECTIVE_FOV_MIN_DEG = 8.0f;
const float PERSPECTIVE_FOV_MAX_DEG = 85.0f;
const float PERSPECTIVE_FOV_STEP_DEG = 2.5f;
const float CAMERA_TILT_MIN = 0.45f;
const float CAMERA_TILT_MAX = 1.65f;
const float Z_HEIGHT_MULTIPLIER = 4.32f;
const float Z_HEIGHT_SCALE_3D = 0.035f;
const float NIGHT_EMISSION_MULT = 3.5f;

const int UDP_CONTROL_PORT = 5005;

// --- Tree params ---
const bool ENABLE_TREES_ON_PARK = true;
const float TREE_TRUNK_RADIUS = 0.055f;
const float TREE_TRUNK_HEIGHT = 0.42f;
const float TREE_CANOPY_RADIUS = 0.24f;
const float TREE_CANOPY_VERTICAL_RADIUS = 0.30f;
const float TREE_CANOPY_CENTER_Y_OFFSET = 0.58f;
const int TREE_TRUNK_SEGMENTS = 10;
const int TREE_CANOPY_LAT_STEPS = 6;
const int TREE_CANOPY_LON_STEPS = 10;

const float OBJ_TARGET_HEIGHT_MULT = 1.0f;
const float OBJ_TARGET_RADIUS_MULT = 1.0f;

const float AUTOEXPOSURE_TARGET_LUMA = 0.42f;
const float AUTOEXPOSURE_MIN = 0.20f;
const float AUTOEXPOSURE_MAX = 8.00f;
const float AUTOEXPOSURE_ADAPT = 0.08f;

const int START_YEAR   = 2026;
const int START_MONTH  = 4;
const int START_DAY    = 9;
const int START_HOUR   = 9;
const int START_MINUTE = 0;
const int START_SECOND = 0;

const int VIDEO_FPS = 30;
const int VIDEO_SIM_MINUTES_PER_FRAME = 1;
const bool WRITE_TIMELAPSE_VIDEO = true;

const double HUNGER_PER_HOUR = 4.0;
const double BLADDER_PER_HOUR = 1.8;
const double ENERGY_GAIN_SLEEP_PER_HOUR = 10.0;
const double ENERGY_LOSS_WORK_PER_HOUR = 5.0;
const double ENERGY_LOSS_LEISURE_PER_HOUR = 4.0;
const double ENERGY_LOSS_IDLE_PER_HOUR = 2.0;
const double SOCIAL_GAIN_RELAX_PER_HOUR = 1.5;
const double SOCIAL_LOSS_NORMAL_PER_HOUR = 1.0;

const float SUN_ROTATION_OFFSET_DEG = -90.0f;

enum Terrain {
    VOID_TERRAIN = 0, ASPHALT, WALKABLE, WALL, PARK, WORK, WC, LIVING, BEDROOM, KITCHEN, SEAWATER, SCHOOL, HIGHSCHOOL, UNIVERSITY
};

typedef pair<int,int> IPoint;

static unordered_map<string, vector<IPoint>> pathCache;
static mt19937 rng((unsigned int)time(nullptr));

string makeKey(const IPoint &start, const IPoint &goal) {
    return to_string(start.first) + "," + to_string(start.second) + "->" + to_string(goal.first) + "," + to_string(goal.second);
}
inline int manhattan(const IPoint &a, const IPoint &b) { return abs(a.first - b.first) + abs(a.second - b.second); }
double frand01() { return uniform_real_distribution<double>(0.0, 1.0)(rng); }
int irand(int a, int b) { return uniform_int_distribution<int>(a, b)(rng); }
template<typename T> T clampT(const T &v, const T &lo, const T &hi) { return max(lo, min(hi, v)); }
long long nowMs() { return (long long)getTickCount() * 1000 / getTickFrequency(); }

static time_t makeStartTimestamp(int year, int month, int day, int hour, int minute, int second) {
    std::tm tmv{};
    tmv.tm_year = year - 1900;
    tmv.tm_mon  = month - 1;
    tmv.tm_mday = day;
    tmv.tm_hour = hour;
    tmv.tm_min  = minute;
    tmv.tm_sec  = second;
    tmv.tm_isdst = -1;
    return mktime(&tmv);
}

static std::tm localTmFromTimeT(time_t t) {
    std::tm out{};
#ifdef _WIN32
    localtime_s(&out, &t);
#else
    localtime_r(&t, &out);
#endif
    return out;
}

int getHour(time_t seconds) {
    std::tm tmv = localTmFromTimeT(seconds);
    return tmv.tm_hour;
}
int getDayIndex(time_t seconds) {
    std::tm tmv = localTmFromTimeT(seconds);
    return tmv.tm_wday;
}
bool isWeekend(time_t seconds) {
    int d = getDayIndex(seconds);
    return d == 0 || d == 6;
}

float lerpAngle(float a, float b, float t) {
    float diff = b - a;
    while(diff > (float)CV_PI) diff -= 2.0f * (float)CV_PI;
    while(diff < -(float)CV_PI) diff += 2.0f * (float)CV_PI;
    return a + diff * t;
}
float normalizeAngle(float a) {
    while(a > (float)CV_PI) a -= 2.0f * (float)CV_PI;
    while(a < -(float)CV_PI) a += 2.0f * (float)CV_PI;
    return a;
}
string secondsToDateTimeStr(time_t seconds) {
    static const vector<string> wd = {"Domingo","Lunes","Martes","Miercoles","Jueves","Viernes","Sabado"};
    std::tm tmv = localTmFromTimeT(seconds);
    ostringstream oss;
    oss << setfill('0')
        << (tmv.tm_year + 1900) << ":"
        << setw(2) << (tmv.tm_mon + 1) << ":"
        << setw(2) << tmv.tm_mday << ":"
        << wd[tmv.tm_wday] << ":"
        << setw(2) << tmv.tm_hour << ":"
        << setw(2) << tmv.tm_min << ":"
        << setw(2) << tmv.tm_sec;
    return oss.str();
}


static string upperStr(string s) {
    for(char &ch : s) ch = (char)toupper((unsigned char)ch);
    return s;
}
static string trimStr(const string& s) {
    size_t a = 0;
    while(a < s.size() && isspace((unsigned char)s[a])) a++;
    size_t b = s.size();
    while(b > a && isspace((unsigned char)s[b-1])) b--;
    return s.substr(a, b - a);
}
static bool startsWith(const string& s, const string& p) {
    return s.size() >= p.size() && equal(p.begin(), p.end(), s.begin());
}

#ifdef __linux__
struct UdpReceiver {
    int sock = -1;
    bool ok = false;
    bool init(int port) {
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        if(sock < 0) return false;
        int flags = fcntl(sock, F_GETFL, 0);
        if(flags >= 0) fcntl(sock, F_SETFL, flags | O_NONBLOCK);
        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons((uint16_t)port);
        if(bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock);
            sock = -1;
            return false;
        }
        ok = true;
        return true;
    }
    void shutdown() {
        if(sock >= 0) close(sock);
        sock = -1;
        ok = false;
    }
    vector<string> recvAll() {
        vector<string> out;
        if(sock < 0) return out;
        while(true) {
            char buf[2048];
            sockaddr_in srcAddr{};
            socklen_t sl = sizeof(srcAddr);
            ssize_t n = recvfrom(sock, buf, sizeof(buf)-1, 0, (sockaddr*)&srcAddr, &sl);
            if(n <= 0) break;
            buf[n] = 0;
            string payload(buf);
            string line;
            stringstream ss(payload);
            while(getline(ss, line)) {
                line = trimStr(line);
                if(!line.empty()) out.push_back(line);
            }
        }
        return out;
    }
};
#else
struct UdpReceiver {
    bool init(int) { return false; }
    void shutdown() {}
    vector<string> recvAll() { return {}; }
};
#endif

bool isWalkable(Terrain t) { return t != VOID_TERRAIN && t != WALL; }

Terrain decodeTerrain(const Vec3b &p) {
    int b = p[0], g = p[1], r = p[2];
    if (r == 255 && g == 0   && b == 255) return ASPHALT;
    if (r == 0   && g == 255 && b == 0)   return WALKABLE;
    if (r == 255 && g == 255 && b == 255) return WALL;
    if (r == 0   && g == 200 && b == 0)   return PARK;
    if (r == 127 && g == 127 && b == 127) return WORK;
    if (r == 255 && g == 0   && b == 0)   return WC;
    if (r == 0   && g == 255 && b == 255) return LIVING;
    if (r == 0   && g == 0   && b == 255) return BEDROOM;
    if (r == 255 && g == 255 && b == 0)   return KITCHEN;
    if (r == 76  && g == 72  && b == 194) return SEAWATER;
    if (r == 213 && g == 132 && b == 50)  return SCHOOL;
    if (r == 142 && g == 95  && b == 48)  return HIGHSCHOOL;
    if (r == 112 && g == 85  && b == 59)  return UNIVERSITY;
    return VOID_TERRAIN;
}

vector<vector<Terrain>> buildTerrain(const Mat &img) {
    int h = img.rows, w = img.cols;
    vector<vector<Terrain>> terrain(h, vector<Terrain>(w, VOID_TERRAIN));
    for(int y = 0; y < h; ++y) for(int x = 0; x < w; ++x) terrain[y][x] = decodeTerrain(img.at<Vec3b>(y, x));
    return terrain;
}
vector<vector<float>> buildZOffsets(const Mat &img, float multiplier) {
    int h = img.rows, w = img.cols;
    vector<vector<float>> zmap(h, vector<float>(w, 0.0f));
    for(int y = 0; y < h; ++y) for(int x = 0; x < w; ++x) {
        int r = img.at<Vec3b>(y, x)[2];
        zmap[y][x] = ((float)r - 127.0f) * multiplier;
    }
    return zmap;
}
float sampleZBilinear(const vector<vector<float>> &zmap, float row, float col) {
    int H = (int)zmap.size(), W = (int)zmap[0].size();
    row = clampT(row, 0.0f, (float)(H - 1));
    col = clampT(col, 0.0f, (float)(W - 1));
    int r0 = (int)floor(row), c0 = (int)floor(col);
    int r1 = min(r0 + 1, H - 1), c1 = min(c0 + 1, W - 1);
    float fr = row - (float)r0, fc = col - (float)c0;
    float z00 = zmap[r0][c0], z01 = zmap[r0][c1], z10 = zmap[r1][c0], z11 = zmap[r1][c1];
    float z0 = z00 * (1.0f - fc) + z01 * fc;
    float z1 = z10 * (1.0f - fc) + z11 * fc;
    return z0 * (1.0f - fr) + z1 * fr;
}

vector<IPoint> aStar(const vector<vector<Terrain>> &terrain, const IPoint &start, const IPoint &goal) {
    if(start == goal) return {};
    int H = (int)terrain.size(), W = (int)terrain[0].size();
    auto keyStr = [&](const IPoint &pt) { return to_string(pt.first) + "," + to_string(pt.second); };
    unordered_map<string, IPoint> cameFrom;
    unordered_map<string, int> gScore, fScore;
    auto cmp = [&](const IPoint &a, const IPoint &b) { return fScore[keyStr(a)] > fScore[keyStr(b)]; };
    priority_queue<IPoint, vector<IPoint>, decltype(cmp)> open(cmp);
    gScore[keyStr(start)] = 0;
    fScore[keyStr(start)] = manhattan(start, goal);
    open.push(start);
    vector<IPoint> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
    while(!open.empty()) {
        IPoint current = open.top(); open.pop();
        if(current == goal) {
            vector<IPoint> path; IPoint cur = current;
            while(cameFrom.count(keyStr(cur))) { path.push_back(cur); cur = cameFrom[keyStr(cur)]; }
            reverse(path.begin(), path.end());
            return path;
        }
        for(const auto &d : dirs) {
            IPoint nb{current.first + d.first, current.second + d.second};
            if(nb.first < 0 || nb.first >= H || nb.second < 0 || nb.second >= W) continue;
            if(!isWalkable(terrain[nb.first][nb.second])) continue;
            int tentative = gScore[keyStr(current)] + 1;
            string nk = keyStr(nb);
            if(!gScore.count(nk) || tentative < gScore[nk]) {
                cameFrom[nk] = current;
                gScore[nk] = tentative;
                fScore[nk] = tentative + manhattan(nb, goal);
                open.push(nb);
            }
        }
    }
    return {};
}
vector<IPoint> getPath(const vector<vector<Terrain>> &terrain, const IPoint &start, const IPoint &goal) {
    string k = makeKey(start, goal);
    auto it = pathCache.find(k);
    if(it != pathCache.end()) return it->second;
    auto path = aStar(terrain, start, goal);
    pathCache[k] = path;
    return path;
}

struct District {
    int id = -1;
    vector<IPoint> walkables, asphalt, walkPaths, parks, works, wcs, livings, bedrooms, kitchens, seawaters, schools, highschools, universities;
};

vector<vector<int>> buildDistrictMap(const vector<vector<Terrain>> &terrain, vector<District> &districts) {
    int H = (int)terrain.size(), W = (int)terrain[0].size();
    vector<vector<int>> comp(H, vector<int>(W, -1));
    vector<IPoint> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
    int nextId = 0;
    for(int y = 0; y < H; ++y) {
        for(int x = 0; x < W; ++x) {
            if(comp[y][x] != -1 || !isWalkable(terrain[y][x])) continue;
            queue<IPoint> q; q.push({y,x}); comp[y][x] = nextId;
            District d; d.id = nextId;
            while(!q.empty()) {
                IPoint p = q.front(); q.pop();
                int r = p.first, c = p.second;
                Terrain t = terrain[r][c];
                d.walkables.push_back(p);
                if(t == ASPHALT) d.asphalt.push_back(p);
                if(t == WALKABLE) d.walkPaths.push_back(p);
                if(t == PARK) d.parks.push_back(p);
                if(t == WORK) d.works.push_back(p);
                if(t == WC) d.wcs.push_back(p);
                if(t == LIVING) d.livings.push_back(p);
                if(t == BEDROOM) d.bedrooms.push_back(p);
                if(t == KITCHEN) d.kitchens.push_back(p);
                if(t == SEAWATER) d.seawaters.push_back(p);
                if(t == SCHOOL) d.schools.push_back(p);
                if(t == HIGHSCHOOL) d.highschools.push_back(p);
                if(t == UNIVERSITY) d.universities.push_back(p);
                for(const auto &dd : dirs) {
                    int ny = r + dd.first, nx = c + dd.second;
                    if(ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                    if(comp[ny][nx] != -1 || !isWalkable(terrain[ny][nx])) continue;
                    comp[ny][nx] = nextId; q.push({ny,nx});
                }
            }
            districts.push_back(d); nextId++;
        }
    }
    return comp;
}
IPoint nearestFromList(const IPoint &from, const vector<IPoint> &list) {
    if(list.empty()) return from;
    return *min_element(list.begin(), list.end(), [&](const IPoint &a, const IPoint &b) { return manhattan(from, a) < manhattan(from, b); });
}
IPoint randomFromList(const vector<IPoint> &list) { if(list.empty()) return {0,0}; return list[irand(0, (int)list.size() - 1)]; }
IPoint nearestNearHome(const IPoint &homeCell, const vector<IPoint> &list) { return nearestFromList(homeCell, list); }
IPoint nearestBathroomFromCurrent(const IPoint &currentCell, const vector<IPoint> &list) { return nearestFromList(currentCell, list); }

struct SmoothMover {
    IPoint cell = {0,0}, moveFromCell = {0,0}, moveToCell = {0,0};
    Point2f renderCell = Point2f(0,0);
    float moveProgress = 1.0f;
    float moveSpeed = 0.22f;
    bool canStartNewStep() const { return moveProgress >= 1.0f; }
    void initAt(const IPoint &start, float spd) {
        cell = start; moveFromCell = start; moveToCell = start;
        renderCell = Point2f((float)start.second, (float)start.first);
        moveProgress = 1.0f; moveSpeed = spd;
    }
    void startMoveTo(const IPoint &next) { moveFromCell = cell; moveToCell = next; moveProgress = 0.0f; }
    void update() {
        if(moveProgress >= 1.0f) { renderCell = Point2f((float)cell.second, (float)cell.first); return; }
        moveProgress += moveSpeed;
        if(moveProgress >= 1.0f) { moveProgress = 1.0f; cell = moveToCell; }
        float t = clampT(moveProgress, 0.0f, 1.0f);
        float rr = (float)moveFromCell.first  + (float)(moveToCell.first  - moveFromCell.first)  * t;
        float cc = (float)moveFromCell.second + (float)(moveToCell.second - moveFromCell.second) * t;
        renderCell = Point2f(cc, rr);
    }
};

enum GoalType {
    GOAL_SLEEP, GOAL_BREAKFAST, GOAL_WORK, GOAL_LUNCH, GOAL_DINNER, GOAL_BATHROOM,
    GOAL_RELAX_HOME, GOAL_PARK, GOAL_SEA_BATH, GOAL_SCHOOL, GOAL_HIGHSCHOOL, GOAL_UNIVERSITY, GOAL_RETIRED_BEACH
};
enum DayRole { ROLE_CHILD_SCHOOL, ROLE_TEEN_HIGHSCHOOL, ROLE_YOUNG_UNIVERSITY, ROLE_ADULT_WORK, ROLE_RETIRED };

struct Persona {
    int id = 0, age = 0;
    SmoothMover mover;
    int homeDistrict = -1;
    IPoint bedroom, workplace, studyplace;
    vector<IPoint> localKitchens, localLivings, localWCs, localParks, localSea;
    vector<IPoint> path;
    IPoint target;
    GoalType currentGoal = GOAL_RELAX_HOME;
    string currentGoalLabel = "Casa";
    double hunger = 20.0, bladder = 10.0, energy = 80.0, social = 60.0;
    time_t stayUntilTime = 0;
    int lastMealHour = -10, lastBathroomHour = -10;
    double moveSmoothing = 0.22;
    int colorVariant = 0;
    bool maleStyle = true;
    float facingAngle = 0.0f, desiredFacingAngle = 0.0f;
    bool returningFromBathroom = false;
    GoalType resumeGoalAfterBathroom = GOAL_RELAX_HOME;
    IPoint resumeTargetAfterBathroom = {0,0};

    IPoint cell() const { return mover.cell; }
    Point2f renderCell() const { return mover.renderCell; }

    void init(int pid, const IPoint &start) {
        id = pid; colorVariant = pid % 8; maleStyle = (pid % 2 == 0); moveSmoothing = 0.14 + frand01() * 0.08;
        mover.initAt(start, (float)moveSmoothing); target = start; facingAngle = (float)(frand01() * CV_PI * 2.0); desiredFacingAngle = facingAngle;
    }
    DayRole getRole() const {
        if(age >= 1 && age <= 12) return ROLE_CHILD_SCHOOL;
        if(age >= 13 && age <= 18) return ROLE_TEEN_HIGHSCHOOL;
        if(age >= 19 && age <= 22) return ROLE_YOUNG_UNIVERSITY;
        if(age >= 23 && age <= 65) return ROLE_ADULT_WORK;
        return ROLE_RETIRED;
    }
    void updateNeeds(time_t /*curTime*/, double deltaSimSeconds) {
        double hours = deltaSimSeconds / 3600.0;
        hunger += HUNGER_PER_HOUR * hours;
        bladder += BLADDER_PER_HOUR * hours;
        if(currentGoal == GOAL_SLEEP) {
            energy += ENERGY_GAIN_SLEEP_PER_HOUR * hours;
        } else if(currentGoal == GOAL_WORK || currentGoal == GOAL_SCHOOL || currentGoal == GOAL_HIGHSCHOOL || currentGoal == GOAL_UNIVERSITY) {
            energy -= ENERGY_LOSS_WORK_PER_HOUR * hours;
        } else if(currentGoal == GOAL_PARK || currentGoal == GOAL_SEA_BATH || currentGoal == GOAL_RETIRED_BEACH) {
            energy -= ENERGY_LOSS_LEISURE_PER_HOUR * hours;
        } else {
            energy -= ENERGY_LOSS_IDLE_PER_HOUR * hours;
        }
        if(currentGoal == GOAL_RELAX_HOME || currentGoal == GOAL_PARK || currentGoal == GOAL_SEA_BATH || currentGoal == GOAL_RETIRED_BEACH) {
            social += SOCIAL_GAIN_RELAX_PER_HOUR * hours;
        } else {
            social -= SOCIAL_LOSS_NORMAL_PER_HOUR * hours;
        }
        hunger = clampT(hunger, 0.0, 100.0);
        bladder = clampT(bladder, 0.0, 100.0);
        energy = clampT(energy, 0.0, 100.0);
        social = clampT(social, 0.0, 100.0);
    }
    GoalType chooseRoutineGoal(time_t curTime) {
        int hour = getHour(curTime); bool weekend = isWeekend(curTime); DayRole role = getRole();
        if((hour >= 23 || hour < 7) && energy < 85.0) return GOAL_SLEEP;
        if(energy < 10.0) return GOAL_SLEEP;
        if(hour >= 7 && hour < 9) {
            if(lastMealHour != hour && hunger > 18.0) return GOAL_BREAKFAST;
            return GOAL_RELAX_HOME;
        }
        if(!weekend && hour >= 9 && hour < 13) {
            switch(role) {
                case ROLE_CHILD_SCHOOL: return GOAL_SCHOOL;
                case ROLE_TEEN_HIGHSCHOOL: return GOAL_HIGHSCHOOL;
                case ROLE_YOUNG_UNIVERSITY: return GOAL_UNIVERSITY;
                case ROLE_ADULT_WORK: return GOAL_WORK;
                case ROLE_RETIRED: return GOAL_RETIRED_BEACH;
            }
        }
        if(hour >= 13 && hour < 15) {
            if(lastMealHour != hour && hunger > 28.0) return GOAL_LUNCH;
            return GOAL_RELAX_HOME;
        }
        if(!weekend && hour >= 15 && hour < 19) {
            switch(role) {
                case ROLE_CHILD_SCHOOL: return GOAL_SCHOOL;
                case ROLE_TEEN_HIGHSCHOOL: return GOAL_HIGHSCHOOL;
                case ROLE_YOUNG_UNIVERSITY: return GOAL_UNIVERSITY;
                case ROLE_ADULT_WORK: return GOAL_WORK;
                case ROLE_RETIRED: return GOAL_RETIRED_BEACH;
            }
        }
        if(hour >= 20 && hour < 22) {
            if(lastMealHour != hour && hunger > 35.0) return GOAL_DINNER;
            if(weekend && !localSea.empty() && frand01() < 0.65) return GOAL_SEA_BATH;
            return GOAL_RELAX_HOME;
        }
        if(weekend && hour >= 10 && hour < 19) {
            if(!localSea.empty()) {
                if(social < 75.0) return GOAL_SEA_BATH;
                if(frand01() < 0.70) return GOAL_SEA_BATH;
            }
            if(social < 65.0) return GOAL_PARK;
            return GOAL_RELAX_HOME;
        }
        if(hour >= 18 && hour < 22 && social < 45.0) {
            if(weekend && !localSea.empty()) return GOAL_SEA_BATH;
            return GOAL_PARK;
        }
        return GOAL_RELAX_HOME;
    }
    GoalType chooseGoal(time_t curTime) {
        int hour = getHour(curTime);
        if(bladder > 92.0 && hour != lastBathroomHour) return GOAL_BATHROOM;
        return chooseRoutineGoal(curTime);
    }
    IPoint chooseGoalTarget(GoalType g) {
        switch(g) {
            case GOAL_SLEEP: currentGoalLabel = "Dormir"; return bedroom;
            case GOAL_BREAKFAST: currentGoalLabel = "Desayuno"; return nearestNearHome(bedroom, localKitchens);
            case GOAL_WORK: currentGoalLabel = "Trabajo"; return workplace;
            case GOAL_SCHOOL: currentGoalLabel = "School"; return studyplace;
            case GOAL_HIGHSCHOOL: currentGoalLabel = "HighSchool"; return studyplace;
            case GOAL_UNIVERSITY: currentGoalLabel = "Universidad"; return studyplace;
            case GOAL_RETIRED_BEACH: currentGoalLabel = "Playa"; return !localSea.empty() ? randomFromList(localSea) : bedroom;
            case GOAL_LUNCH: currentGoalLabel = "Comida"; return nearestNearHome(bedroom, localKitchens);
            case GOAL_DINNER: currentGoalLabel = "Cena"; return nearestNearHome(bedroom, localKitchens);
            case GOAL_BATHROOM: currentGoalLabel = "WC"; return nearestBathroomFromCurrent(cell(), localWCs);
            case GOAL_SEA_BATH: currentGoalLabel = "Bano mar"; return !localSea.empty() ? randomFromList(localSea) : bedroom;
            case GOAL_PARK: currentGoalLabel = "Parque"; return !localParks.empty() ? randomFromList(localParks) : bedroom;
            default: currentGoalLabel = "Salon"; return nearestNearHome(bedroom, localLivings);
        }
    }
    void rebuildPath(const vector<vector<Terrain>> &terrain) { path = getPath(terrain, cell(), target); }

    time_t chooseStayDurationSeconds() const {
        switch(currentGoal) {
            case GOAL_SLEEP:         return 20 * 60 + irand(0, 20 * 60);
            case GOAL_BREAKFAST:     return 10 * 60 + irand(0, 8 * 60);
            case GOAL_LUNCH:         return 18 * 60 + irand(0, 12 * 60);
            case GOAL_DINNER:        return 16 * 60 + irand(0, 14 * 60);
            case GOAL_WORK:
            case GOAL_SCHOOL:
            case GOAL_HIGHSCHOOL:
            case GOAL_UNIVERSITY:    return 30 * 60 + irand(0, 30 * 60);
            case GOAL_BATHROOM:      return 3 * 60 + irand(0, 3 * 60);
            case GOAL_RETIRED_BEACH:
            case GOAL_SEA_BATH:      return 20 * 60 + irand(0, 20 * 60);
            case GOAL_PARK:          return 15 * 60 + irand(0, 15 * 60);
            default:                 return 20 * 60 + irand(0, 15 * 60);
        }
    }

    void interactAtDestination(time_t curTime) {
        int hour = getHour(curTime);
        switch(currentGoal) {
            case GOAL_SLEEP:
                energy += 8.0;
                hunger += 1.0;
                bladder += 1.5;
                break;
            case GOAL_BREAKFAST:
            case GOAL_LUNCH:
            case GOAL_DINNER:
                hunger -= 22.0;
                energy += 2.0;
                bladder += 1.2;
                lastMealHour = hour;
                break;
            case GOAL_WORK:
            case GOAL_SCHOOL:
            case GOAL_HIGHSCHOOL:
            case GOAL_UNIVERSITY:
                energy -= 1.2;
                hunger += 1.0;
                bladder += 0.8;
                break;
            case GOAL_BATHROOM:
                bladder -= 75.0;
                lastBathroomHour = hour;
                break;
            case GOAL_RETIRED_BEACH:
            case GOAL_SEA_BATH:
                social += 5.0;
                energy -= 1.2;
                hunger += 0.8;
                bladder += 0.5;
                break;
            case GOAL_PARK:
                social += 3.5;
                energy -= 0.8;
                hunger += 0.4;
                bladder += 0.3;
                break;
            default:
                social += 1.5;
                energy += 0.8;
                break;
        }
        hunger = clampT(hunger, 0.0, 100.0);
        bladder = clampT(bladder, 0.0, 100.0);
        energy = clampT(energy, 0.0, 100.0);
        social = clampT(social, 0.0, 100.0);
        stayUntilTime = curTime + chooseStayDurationSeconds();
    }
    void updateMovementOnly() {
        Point2f before = mover.renderCell; mover.update(); Point2f after = mover.renderCell;
        Point2f delta = after - before;
        if(norm(delta) > 0.0005f) desiredFacingAngle = atan2(delta.y, delta.x);
        facingAngle = lerpAngle(facingAngle, desiredFacingAngle, 0.18f);
    }
    void stepDecision(const vector<vector<Terrain>> &terrain, time_t curTime, double deltaSimSeconds) {
        updateNeeds(curTime, deltaSimSeconds);
        if(!mover.canStartNewStep()) return;
        if(curTime < stayUntilTime) return;
        if(currentGoal == GOAL_BATHROOM && path.empty()) {
            interactAtDestination(curTime);
            if(returningFromBathroom) {
                currentGoal = resumeGoalAfterBathroom;
                target = resumeTargetAfterBathroom;
                returningFromBathroom = false;
                rebuildPath(terrain);
                stayUntilTime = curTime;
                return;
            }
        }
        GoalType g = chooseGoal(curTime);
        IPoint wanted;
        if(g == GOAL_BATHROOM) {
            GoalType afterWC = chooseRoutineGoal(curTime);
            IPoint afterWCTarget = chooseGoalTarget(afterWC);
            if(currentGoal != GOAL_BATHROOM) {
                returningFromBathroom = true;
                resumeGoalAfterBathroom = afterWC;
                resumeTargetAfterBathroom = afterWCTarget;
            }
            wanted = chooseGoalTarget(GOAL_BATHROOM);
        } else {
            returningFromBathroom = false;
            wanted = chooseGoalTarget(g);
        }
        if(g != currentGoal || wanted != target || path.empty()) {
            currentGoal = g;
            target = wanted;
            rebuildPath(terrain);
        }
        if(!path.empty()) {
            IPoint next = path.front();
            path.erase(path.begin());
            Point2f delta((float)(next.second - cell().second), (float)(next.first - cell().first));
            if(norm(delta) > 0.001f) desiredFacingAngle = atan2(delta.y, delta.x);
            mover.startMoveTo(next);
        } else {
            interactAtDestination(curTime);
        }
    }
};

struct Player {
    SmoothMover mover;
    float facingAngle = 0.0f, desiredFacingAngle = 0.0f;
    void init(const IPoint &start) { mover.initAt(start, 0.18f); }
    IPoint cell() const { return mover.cell; }
    Point2f renderCell() const { return mover.renderCell; }
    void update() {
        Point2f before = mover.renderCell; mover.update(); Point2f after = mover.renderCell;
        Point2f delta = after - before;
        if(norm(delta) > 0.0005f) desiredFacingAngle = atan2(delta.y, delta.x);
        facingAngle = lerpAngle(facingAngle, desiredFacingAngle, 0.22f);
    }
    bool tryMove(int dr, int dc, const vector<vector<Terrain>> &terrain) {
        if(!mover.canStartNewStep()) return false;
        IPoint cur = cell(); IPoint nxt = {cur.first + dr, cur.second + dc};
        int H = (int)terrain.size(), W = (int)terrain[0].size();
        if(nxt.first < 0 || nxt.first >= H || nxt.second < 0 || nxt.second >= W) return false;
        if(!isWalkable(terrain[nxt.first][nxt.second])) return false;
        Point2f delta((float)dc, (float)dr);
        if(norm(delta) > 0.001f) desiredFacingAngle = atan2(delta.y, delta.x);
        mover.startMoveTo(nxt); return true;
    }
};

struct Camera2D {
    float centerX = 0.0f, centerY = 0.0f, zoom = 1.0f;
    float targetCenterX = 0.0f, targetCenterY = 0.0f, targetZoom = 1.0f;
    float manualZoom = CAMERA_NPC_ZOOM_FACTOR;
    float orbitYaw = (float)(45.0 * CV_PI / 180.0), targetOrbitYaw = (float)(45.0 * CV_PI / 180.0);
    float tilt = 1.0f, targetTilt = 1.0f;
    float perspectiveFovDeg = 14.0f;
    bool conicPerspective = false;
    int focusedNpc = 0;
    bool followPlayer = true;
    bool autoReturnToPlayerWhenMoving = false;
    void changeZoom(float delta) { manualZoom = clampT(manualZoom + delta, CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX); targetZoom = manualZoom; }
    void changeOrbitDeg(float deltaDeg) {
        float delta = (float)(deltaDeg * CV_PI / 180.0);
        targetOrbitYaw = normalizeAngle(targetOrbitYaw + delta);
        orbitYaw = targetOrbitYaw;
    }
    void changeTilt(float delta) {
        targetTilt = clampT(targetTilt + delta, CAMERA_TILT_MIN, CAMERA_TILT_MAX);
        tilt = targetTilt;
    }
    void toggleProjectionMode() { conicPerspective = !conicPerspective; }
    void changePerspectiveFov(float deltaDeg) { perspectiveFovDeg = clampT(perspectiveFovDeg + deltaDeg, PERSPECTIVE_FOV_MIN_DEG, PERSPECTIVE_FOV_MAX_DEG); }
    void nextNpcFocus(const vector<Persona> &agents) { if(agents.empty()) return; focusedNpc = (focusedNpc + 1) % (int)agents.size(); followPlayer = false; }
    void prevNpcFocus(const vector<Persona> &agents) { if(agents.empty()) return; focusedNpc = (focusedNpc - 1 + (int)agents.size()) % (int)agents.size(); followPlayer = false; }
    void setZoom(float v) { manualZoom = clampT(v, CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX); targetZoom = manualZoom; zoom = manualZoom; }
    void setTilt(float v) { targetTilt = clampT(v, CAMERA_TILT_MIN, CAMERA_TILT_MAX); tilt = targetTilt; }
    void setOrbitDeg(float deg) { targetOrbitYaw = normalizeAngle((float)(deg * CV_PI / 180.0)); orbitYaw = targetOrbitYaw; }
    float orbitDeg() const { return orbitYaw * 180.0f / (float)CV_PI; }
    void updateFollowPoint(float rx, float ry) {
        targetCenterX = rx; targetCenterY = ry; targetZoom = manualZoom;
        centerX += (targetCenterX - centerX) * CAMERA_CENTER_SMOOTH;
        centerY += (targetCenterY - centerY) * CAMERA_CENTER_SMOOTH;
        zoom    += (targetZoom    - zoom)    * CAMERA_ZOOM_SMOOTH;
        orbitYaw = lerpAngle(orbitYaw, targetOrbitYaw, CAMERA_ANGLE_SMOOTH);
        tilt += (targetTilt - tilt) * CAMERA_ANGLE_SMOOTH;
    }
    void updatePlayerOrCinematic(const Player &player, const vector<Persona> &agents, long long nowRealMs, long long lastPlayerInputMs) {
        bool playerRecentlyMoved = (nowRealMs - lastPlayerInputMs) < PLAYER_IDLE_FALLBACK_MS;
        if (followPlayer) {
            updateFollowPoint(player.renderCell().x, player.renderCell().y);
            return;
        }
        if (agents.empty()) {
            updateFollowPoint(player.renderCell().x, player.renderCell().y);
            return;
        }
        if (autoReturnToPlayerWhenMoving && playerRecentlyMoved) {
            followPlayer = true;
            updateFollowPoint(player.renderCell().x, player.renderCell().y);
            return;
        }
        const Persona &a = agents[focusedNpc];
        updateFollowPoint(a.renderCell().x, a.renderCell().y);
    }
};

struct Camera3D { float3 pos, target, up; float fovY; float orthoScale; int projectionMode; };
static inline float worldHeightAtCell(const vector<vector<float>>& zmap, float row, float col) { return sampleZBilinear(zmap, row, col) * Z_HEIGHT_SCALE_3D; }
static inline float3 cellToWorld(float row, float col, const vector<vector<float>>& zmap) { return make_f3(col * TILE_WORLD, worldHeightAtCell(zmap, row, col), row * TILE_WORLD); }
static Camera3D buildCamera3D(const Camera2D& cam2, const vector<vector<float>>& zmap) {
    Camera3D c{};
    float3 center = cellToWorld(cam2.centerY, cam2.centerX, zmap);
    center.x += TILE_WORLD * 0.5f;
    center.z += TILE_WORLD * 0.5f;
    float yaw = cam2.orbitYaw;
    float pitch = 0.615f + (cam2.tilt - 1.0f) * 0.35f;
    float3 dir;
    dir.x = cosf(pitch) * cosf(yaw);
    dir.y = sinf(pitch);
    dir.z = cosf(pitch) * sinf(yaw);
    c.target = center;
    c.up = make_f3(0, 1, 0);
    c.projectionMode = cam2.conicPerspective ? 1 : 0;
    if (cam2.conicPerspective) {
        float distance = 48.0f / cam2.zoom;
        c.pos = center + dir * distance;
        c.fovY = cam2.perspectiveFovDeg;
        c.orthoScale = 0.0f;
    } else {
        float distance = 64.0f;
        c.pos = center + dir * distance;
        c.fovY = cam2.perspectiveFovDeg;
        c.orthoScale = 18.0f / cam2.zoom;
    }
    return c;
}

struct Tri {
    float3 v0, v1, v2;
    float3 n0, n1, n2;
    float3 albedo;
    float3 specular;
    float  glossiness;
    int materialType;
};
struct AABB { float3 bmin, bmax; };
static inline AABB aabb_empty(){ AABB b; b.bmin = make_f3( 1e30f, 1e30f, 1e30f); b.bmax = make_f3(-1e30f,-1e30f,-1e30f); return b; }
static inline void aabb_extend(AABB& b, const float3& p){ b.bmin = min3(b.bmin, p); b.bmax = max3(b.bmax, p); }
static inline void aabb_extend(AABB& b, const AABB& o){ b.bmin = min3(b.bmin, o.bmin); b.bmax = max3(b.bmax, o.bmax); }
static inline AABB tri_aabb(const Tri& t){ AABB b = aabb_empty(); aabb_extend(b, t.v0); aabb_extend(b, t.v1); aabb_extend(b, t.v2); return b; }
static inline float3 tri_centroid(const Tri& t){ return (t.v0 + t.v1 + t.v2) / 3.0f; }

struct BVHNode {
    float3 bmin, bmax;
    int left, right, start, count;
};

static int build_bvh_recursive(std::vector<BVHNode>& nodes, std::vector<int>& triIdx, const std::vector<Tri>& tris, int start, int end){
    BVHNode node{};
    AABB box = aabb_empty();
    AABB cbox = aabb_empty();
    for(int i=start;i<end;i++){
        const Tri& t = tris[triIdx[i]];
        aabb_extend(box, tri_aabb(t));
        aabb_extend(cbox, tri_centroid(t));
    }
    node.bmin = box.bmin; node.bmax = box.bmax; node.left = node.right = -1; node.start = start; node.count = end - start;
    int myIndex = (int)nodes.size(); nodes.push_back(node);
    const int LEAF_N = 4; if (end - start <= LEAF_N) return myIndex;
    float3 ext = cbox.bmax - cbox.bmin;
    int axis = 0;
    if (ext.y > ext.x && ext.y > ext.z) axis = 1;
    else if (ext.z > ext.x && ext.z > ext.y) axis = 2;
    int mid = (start + end) / 2;
    std::nth_element(triIdx.begin()+start, triIdx.begin()+mid, triIdx.begin()+end, [&](int a, int b){
        float3 ca = tri_centroid(tris[a]), cb = tri_centroid(tris[b]);
        if(axis==0) return ca.x < cb.x;
        if(axis==1) return ca.y < cb.y;
        return ca.z < cb.z;
    });
    int L = build_bvh_recursive(nodes, triIdx, tris, start, mid);
    int R = build_bvh_recursive(nodes, triIdx, tris, mid, end);
    nodes[myIndex].left = L; nodes[myIndex].right = R; nodes[myIndex].start = -1; nodes[myIndex].count = 0;
    return myIndex;
}

static inline void addTri(vector<Tri>& tris, const float3& a, const float3& b, const float3& c, const float3& n, const float3& albedo, int mat=0) {
    Tri t{}; t.v0=a; t.v1=b; t.v2=c; t.n0=t.n1=t.n2=n; t.albedo=albedo; t.specular=make_f3(0,0,0); t.glossiness=8.0f; t.materialType=mat; tris.push_back(t);
}
static inline void addQuad(vector<Tri>& tris, const float3& a, const float3& b, const float3& c, const float3& d, const float3& albedo, int mat=0) {
    float3 n = norm3(cross3(b-a, c-a));
    addTri(tris, a,b,c,n,albedo,mat);
    addTri(tris, a,c,d,n,albedo,mat);
}
static inline void addBoxMesh(vector<Tri>& tris, const float3& mn, const float3& mx, const float3& albedo, int mat=0) {
    float3 p000 = make_f3(mn.x,mn.y,mn.z), p001 = make_f3(mn.x,mn.y,mx.z);
    float3 p010 = make_f3(mn.x,mx.y,mn.z), p011 = make_f3(mn.x,mx.y,mx.z);
    float3 p100 = make_f3(mx.x,mn.y,mn.z), p101 = make_f3(mx.x,mn.y,mx.z);
    float3 p110 = make_f3(mx.x,mx.y,mn.z), p111 = make_f3(mx.x,mx.y,mx.z);
    addQuad(tris, p010,p110,p111,p011, albedo, mat);
    addQuad(tris, p000,p001,p101,p100, albedo * 0.75f, mat);
    addQuad(tris, p000,p100,p110,p010, albedo * 0.78f, mat);
    addQuad(tris, p100,p101,p111,p110, albedo * 0.72f, mat);
    addQuad(tris, p101,p001,p011,p111, albedo * 0.82f, mat);
    addQuad(tris, p001,p000,p010,p011, albedo * 0.68f, mat);
}

static inline void addCylinderMesh(vector<Tri>& tris, const float3& center, float radius, float height, int segments, const float3& albedo, int mat=0) {
    segments = max(segments, 3);
    float y0 = center.y;
    float y1 = center.y + height;
    for (int i = 0; i < segments; ++i) {
        float a0 = 2.0f * (float)M_PI * (float)i / (float)segments;
        float a1 = 2.0f * (float)M_PI * (float)(i + 1) / (float)segments;

        float3 b0 = make_f3(center.x + cosf(a0) * radius, y0, center.z + sinf(a0) * radius);
        float3 b1 = make_f3(center.x + cosf(a1) * radius, y0, center.z + sinf(a1) * radius);
        float3 t0 = make_f3(center.x + cosf(a0) * radius, y1, center.z + sinf(a0) * radius);
        float3 t1 = make_f3(center.x + cosf(a1) * radius, y1, center.z + sinf(a1) * radius);

        float3 n0 = norm3(make_f3(cosf(a0), 0.0f, sinf(a0)));
        float3 n1 = norm3(make_f3(cosf(a1), 0.0f, sinf(a1)));

        Tri s1{};
        s1.v0 = b0; s1.v1 = t0; s1.v2 = t1;
        s1.n0 = n0; s1.n1 = n0; s1.n2 = n1;
        s1.albedo = albedo; s1.specular = make_f3(0,0,0); s1.glossiness = 8.0f; s1.materialType = mat;
        tris.push_back(s1);

        Tri s2{};
        s2.v0 = b0; s2.v1 = t1; s2.v2 = b1;
        s2.n0 = n0; s2.n1 = n1; s2.n2 = n1;
        s2.albedo = albedo * 0.94f; s2.specular = make_f3(0,0,0); s2.glossiness = 8.0f; s2.materialType = mat;
        tris.push_back(s2);

        addTri(tris, make_f3(center.x, y1, center.z), t1, t0, make_f3(0,1,0), albedo * 0.88f, mat);
        addTri(tris, make_f3(center.x, y0, center.z), b0, b1, make_f3(0,-1,0), albedo * 0.72f, mat);
    }
}

static inline void addEllipsoidMesh(vector<Tri>& tris, const float3& center, float radiusXZ, float radiusY, int latSteps, int lonSteps, const float3& albedo, int mat=0) {
    latSteps = max(latSteps, 3);
    lonSteps = max(lonSteps, 4);
    for (int lat = 0; lat < latSteps; ++lat) {
        float v0 = (float)lat / (float)latSteps;
        float v1 = (float)(lat + 1) / (float)latSteps;
        float phi0 = v0 * (float)M_PI;
        float phi1 = v1 * (float)M_PI;

        for (int lon = 0; lon < lonSteps; ++lon) {
            float u0 = (float)lon / (float)lonSteps;
            float u1 = (float)(lon + 1) / (float)lonSteps;
            float th0 = u0 * 2.0f * (float)M_PI;
            float th1 = u1 * 2.0f * (float)M_PI;

            auto ep = [&](float phi, float th) {
                float sx = sinf(phi) * cosf(th);
                float sy = cosf(phi);
                float sz = sinf(phi) * sinf(th);
                return make_f3(center.x + sx * radiusXZ, center.y + sy * radiusY, center.z + sz * radiusXZ);
            };
            auto en = [&](float phi, float th) {
                float sx = sinf(phi) * cosf(th) / max(radiusXZ, 1e-5f);
                float sy = cosf(phi) / max(radiusY, 1e-5f);
                float sz = sinf(phi) * sinf(th) / max(radiusXZ, 1e-5f);
                return norm3(make_f3(sx, sy, sz));
            };

            float3 p00 = ep(phi0, th0);
            float3 p10 = ep(phi1, th0);
            float3 p11 = ep(phi1, th1);
            float3 p01 = ep(phi0, th1);

            float3 n00 = en(phi0, th0);
            float3 n10 = en(phi1, th0);
            float3 n11 = en(phi1, th1);
            float3 n01 = en(phi0, th1);

            if (lat > 0) {
                Tri t1{};
                t1.v0 = p00; t1.v1 = p10; t1.v2 = p11;
                t1.n0 = n00; t1.n1 = n10; t1.n2 = n11;
                t1.albedo = albedo; t1.specular = make_f3(0,0,0); t1.glossiness = 8.0f; t1.materialType = mat;
                tris.push_back(t1);
            }
            if (lat < latSteps - 1) {
                Tri t2{};
                t2.v0 = p00; t2.v1 = p11; t2.v2 = p01;
                t2.n0 = n00; t2.n1 = n11; t2.n2 = n01;
                t2.albedo = albedo * 0.96f; t2.specular = make_f3(0,0,0); t2.glossiness = 8.0f; t2.materialType = mat;
                tris.push_back(t2);
            }
        }
    }
}

static inline void addTreeMesh(vector<Tri>& tris, const float3& baseCenter) {
    const float3 trunkColor = make_f3(0.40f, 0.25f, 0.10f);
    const float3 canopyColor = make_f3(0.14f, 0.56f, 0.16f);
    addCylinderMesh(tris, baseCenter, TREE_TRUNK_RADIUS, TREE_TRUNK_HEIGHT, TREE_TRUNK_SEGMENTS, trunkColor, 0);
    float3 canopyCenter = make_f3(baseCenter.x, baseCenter.y + TREE_CANOPY_CENTER_Y_OFFSET, baseCenter.z);
    addEllipsoidMesh(tris, canopyCenter, TREE_CANOPY_RADIUS, TREE_CANOPY_VERTICAL_RADIUS, TREE_CANOPY_LAT_STEPS, TREE_CANOPY_LON_STEPS, canopyColor, 0);
}

static inline float3 terrainAlbedo(Terrain t) {
    switch(t) {
        case ASPHALT:    return make_f3(0.38f, 0.34f, 0.42f);
        case WALKABLE:   return make_f3(0.53f, 0.53f, 0.53f);
        case WALL:       return make_f3(0.88f, 0.88f, 0.88f);
        case PARK:       return make_f3(0.16f, 0.58f, 0.22f);
        case WORK:       return make_f3(0.52f, 0.52f, 0.52f);
        case WC:         return make_f3(0.23f, 0.25f, 0.86f);
        case LIVING:     return make_f3(0.20f, 0.84f, 0.84f);
        case BEDROOM:    return make_f3(0.86f, 0.31f, 0.31f);
        case KITCHEN:    return make_f3(0.87f, 0.87f, 0.26f);
        case SEAWATER:   return make_f3(0.78f, 0.32f, 0.12f);
        case SCHOOL:     return make_f3(0.82f, 0.58f, 0.25f);
        case HIGHSCHOOL: return make_f3(0.56f, 0.38f, 0.19f);
        case UNIVERSITY: return make_f3(0.44f, 0.33f, 0.23f);
        default:         return make_f3(0.1f, 0.1f, 0.1f);
    }
}
static inline void buildStaticTerrainMesh(const vector<vector<Terrain>>& terrain, const vector<vector<float>>& zmap, vector<Tri>& tris) {
    int H = (int)terrain.size(), W = (int)terrain[0].size();
    for(int r=0;r<H;r++) for(int c=0;c<W;c++) {
        Terrain t = terrain[r][c];
        if(t == VOID_TERRAIN) continue;
        float baseY = worldHeightAtCell(zmap, (float)r, (float)c);
        float x0 = c * TILE_WORLD, z0 = r * TILE_WORLD;
        float x1 = x0 + TILE_WORLD, z1 = z0 + TILE_WORLD;
        float h = (t == WALL) ? WALL_HEIGHT_3D : FLOOR_THICKNESS_3D;
        float y0 = baseY;
        float y1 = baseY + h + ((t == WALL) ? WALL_HEIGHT_OFFSET_3D : 0.0f);
        int mat = 0;
        if (t == SEAWATER) mat = 1;
        if (t == KITCHEN || t == BEDROOM || t == LIVING || t == WC) mat = 3;
        addBoxMesh(tris, make_f3(x0, y0, z0), make_f3(x1, y1, z1), terrainAlbedo(t), mat);

        if (ENABLE_TREES_ON_PARK && t == PARK) {
            float3 treeBase = make_f3((x0 + x1) * 0.5f, y1, (z0 + z1) * 0.5f);
            addTreeMesh(tris, treeBase);
        }
    }
}

struct MaterialCPU {
    float3 Kd = make_f3(0.75f, 0.75f, 0.75f);
    float3 Ks = make_f3(0.0f, 0.0f, 0.0f);
    float Ns = 32.0f;
};

static std::string trim(const std::string& s){
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char)s[a])) a++;
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char)s[b-1])) b--;
    return s.substr(a, b - a);
}
static std::string get_directory_part(const std::string& path){
    size_t p1 = path.find_last_of('/');
    size_t p2 = path.find_last_of('\\');
    size_t p = std::string::npos;
    if (p1 == std::string::npos) p = p2;
    else if (p2 == std::string::npos) p = p1;
    else p = std::max(p1, p2);
    if (p == std::string::npos) return "";
    return path.substr(0, p + 1);
}
static std::string join_path_simple(const std::string& baseDir, const std::string& fileName){
    if (baseDir.empty()) return fileName;
    return baseDir + fileName;
}
static bool load_mtl(const std::string& mtlPath, std::unordered_map<std::string, MaterialCPU>& materials){
    std::ifstream f(mtlPath);
    if (!f) {
        std::cerr << "Warning: could not open MTL file: " << mtlPath << "\n";
        return false;
    }
    std::string line;
    std::string currentName;
    MaterialCPU currentMat;
    bool hasCurrent = false;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "newmtl") {
            if (hasCurrent && !currentName.empty()) materials[currentName] = currentMat;
            currentName.clear();
            ss >> currentName;
            currentMat = MaterialCPU{};
            hasCurrent = true;
        } else if (tag == "Kd") {
            float r=0.75f, g=0.75f, b=0.75f;
            ss >> r >> g >> b;
            currentMat.Kd = clamp01(make_f3(r,g,b));
        } else if (tag == "Ks") {
            float r=0.0f, g=0.0f, b=0.0f;
            ss >> r >> g >> b;
            currentMat.Ks = clamp01(make_f3(r,g,b));
        } else if (tag == "Ns") {
            float ns = 32.0f;
            ss >> ns;
            currentMat.Ns = fmaxf(1.0f, ns);
        }
    }
    if (hasCurrent && !currentName.empty()) materials[currentName] = currentMat;
    return true;
}
static bool parse_face_vertex(const std::string& tok, int& vi, int& ni){
    vi = 0; ni = 0;
    int a=0,b=0,c=0; char ch;
    std::stringstream ss(tok);
    if (!(ss >> a)) return false;
    vi = a;
    if (ss.peek() == '/') {
        ss >> ch;
        if (ss.peek() == '/') {
            ss >> ch;
            if (ss >> c) ni = c;
        } else {
            if (ss >> b) {
                if (ss.peek() == '/') {
                    ss >> ch;
                    if (ss >> c) ni = c;
                }
            }
        }
    }
    return true;
}
static bool load_obj_with_mtl(const std::string& path, std::vector<Tri>& outTris){
    std::ifstream f(path);
    if (!f) return false;
    std::vector<float3> V;
    std::vector<float3> N;
    std::unordered_map<std::string, MaterialCPU> materials;
    MaterialCPU currentMaterial;
    currentMaterial.Kd = make_f3(0.75f, 0.75f, 0.75f);
    currentMaterial.Ks = make_f3(0.0f, 0.0f, 0.0f);
    currentMaterial.Ns = 32.0f;
    const std::string baseDir = get_directory_part(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x,y,z; ss >> x >> y >> z;
            V.push_back(make_f3(x,y,z));
        } else if (tag == "vn") {
            float x,y,z; ss >> x >> y >> z;
            N.push_back(norm3(make_f3(x,y,z)));
        } else if (tag == "mtllib") {
            std::string mtlFile; ss >> mtlFile;
            if (!mtlFile.empty()) load_mtl(join_path_simple(baseDir, mtlFile), materials);
        } else if (tag == "usemtl") {
            std::string currentMaterialName; ss >> currentMaterialName;
            auto it = materials.find(currentMaterialName);
            if (it != materials.end()) currentMaterial = it->second;
            else currentMaterial = MaterialCPU{};
        } else if (tag == "f") {
            std::vector<int> vis, nis;
            std::string tok;
            while (ss >> tok) {
                int vi=0, ni=0;
                if (!parse_face_vertex(tok, vi, ni)) continue;
                if (vi < 0) vi = (int)V.size() + 1 + vi;
                if (ni < 0) ni = (int)N.size() + 1 + ni;
                vis.push_back(vi - 1);
                nis.push_back(ni - 1);
            }
            if (vis.size() < 3) continue;
            for (size_t i=1; i+1<vis.size(); i++) {
                if (vis[0] < 0 || vis[i] < 0 || vis[i+1] < 0) continue;
                if (vis[0] >= (int)V.size() || vis[i] >= (int)V.size() || vis[i+1] >= (int)V.size()) continue;
                Tri t{};
                t.v0 = V[vis[0]]; t.v1 = V[vis[i]]; t.v2 = V[vis[i+1]];
                bool hasN = (nis[0] >= 0 && nis[i] >= 0 && nis[i+1] >= 0 &&
                            nis[0] < (int)N.size() && nis[i] < (int)N.size() && nis[i+1] < (int)N.size());
                if (hasN) {
                    t.n0 = N[nis[0]]; t.n1 = N[nis[i]]; t.n2 = N[nis[i+1]];
                } else {
                    float3 fn = norm3(cross3(t.v1 - t.v0, t.v2 - t.v0));
                    t.n0 = t.n1 = t.n2 = fn;
                }
                t.albedo = currentMaterial.Kd;
                t.specular = currentMaterial.Ks;
                t.glossiness = fmaxf(2.0f, currentMaterial.Ns);
                t.materialType = 2;
                outTris.push_back(t);
            }
        }
    }
    if (outTris.empty()) {
        std::cerr << "OBJ loaded but no triangles found: " << path << "\n";
        return false;
    }
    return true;
}

struct ObjInstanceTemplate {
    vector<Tri> tris;
    AABB bounds;
    float scale = 1.0f;
    float3 pivotCenterXZ = make_f3(0,0,0);
    float minY = 0.0f;
    AABB localBoundsNormalized;
};

static ObjInstanceTemplate prepareObjTemplate(const vector<Tri>& src) {
    ObjInstanceTemplate T{};
    T.tris = src;
    T.bounds = aabb_empty();
    for (const auto& tri : src) {
        aabb_extend(T.bounds, tri.v0);
        aabb_extend(T.bounds, tri.v1);
        aabb_extend(T.bounds, tri.v2);
    }
    float h = fmaxf(1e-5f, T.bounds.bmax.y - T.bounds.bmin.y);
    float rx = 0.5f * (T.bounds.bmax.x - T.bounds.bmin.x);
    float rz = 0.5f * (T.bounds.bmax.z - T.bounds.bmin.z);
    float r = fmaxf(1e-5f, fmaxf(rx, rz));
    float sH = (HUMAN_HEIGHT_3D * OBJ_TARGET_HEIGHT_MULT) / h;
    float sR = (HUMAN_RADIUS_3D * OBJ_TARGET_RADIUS_MULT) / r;
    T.scale = fminf(sH, sR);
    T.pivotCenterXZ = make_f3((T.bounds.bmin.x + T.bounds.bmax.x) * 0.5f, 0.0f, (T.bounds.bmin.z + T.bounds.bmax.z) * 0.5f);
    T.minY = T.bounds.bmin.y;

    T.localBoundsNormalized = aabb_empty();
    vector<Tri> normalized;
    normalized.reserve(src.size());
    for (const auto& tri : src) {
        Tri t = tri;
        float3* vv[3] = { &t.v0, &t.v1, &t.v2 };
        for(int i=0;i<3;i++) {
            vv[i]->x -= T.pivotCenterXZ.x;
            vv[i]->z -= T.pivotCenterXZ.z;
            vv[i]->y -= T.minY;
            *vv[i] = (*vv[i]) * T.scale;
            aabb_extend(T.localBoundsNormalized, *vv[i]);
        }
        normalized.push_back(t);
    }
    T.tris.swap(normalized);
    return T;
}

static inline float3 rotateY(const float3& p, float ang) {
    float c = cosf(ang), s = sinf(ang);
    return make_f3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}
static inline float3 invRotateY(const float3& p, float ang) {
    return rotateY(p, -ang);
}
static inline float3 transformObjVertex(const float3& v, const ObjInstanceTemplate& T, const float3& worldPos, float yaw) {
    float3 q = v;
    q.x -= T.pivotCenterXZ.x;
    q.z -= T.pivotCenterXZ.z;
    q.y -= T.minY;
    q = q * T.scale;
    q = rotateY(q, yaw);
    return q + worldPos;
}
static inline float3 transformObjNormal(const float3& n, float yaw) {
    return norm3(rotateY(n, yaw));
}

struct GpuScene {
    Tri* d_tris = nullptr;
    int* d_triIdx = nullptr;
    BVHNode* d_nodes = nullptr;
    int triCount = 0;
    int nodeCount = 0;
    int root = -1;
};
static void free_scene(GpuScene& s){
    if (s.d_tris) CUDA_CHECK(cudaFree(s.d_tris));
    if (s.d_triIdx) CUDA_CHECK(cudaFree(s.d_triIdx));
    if (s.d_nodes) CUDA_CHECK(cudaFree(s.d_nodes));
    s = {};
    s.root = -1;
}
static bool upload_scene(const std::vector<Tri>& tris, GpuScene& out){
    free_scene(out);
    if (tris.empty()) {
        out.triCount = 0; out.nodeCount = 0; out.root = -1;
        return true;
    }
    std::vector<int> triIdx(tris.size());
    for (int i=0;i<(int)triIdx.size();i++) triIdx[i]=i;
    std::vector<BVHNode> nodes; nodes.reserve(tris.size()*2);
    int root = build_bvh_recursive(nodes, triIdx, tris, 0, (int)triIdx.size());
    out.triCount = (int)tris.size(); out.nodeCount = (int)nodes.size(); out.root = root;
    CUDA_CHECK(cudaMalloc(&out.d_tris, sizeof(Tri)*out.triCount));
    CUDA_CHECK(cudaMemcpy(out.d_tris, tris.data(), sizeof(Tri)*out.triCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&out.d_triIdx, sizeof(int)*out.triCount));
    CUDA_CHECK(cudaMemcpy(out.d_triIdx, triIdx.data(), sizeof(int)*out.triCount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&out.d_nodes, sizeof(BVHNode)*out.nodeCount));
    CUDA_CHECK(cudaMemcpy(out.d_nodes, nodes.data(), sizeof(BVHNode)*out.nodeCount, cudaMemcpyHostToDevice));
    return true;
}

struct InstanceGPU {
    float3 pos;
    float yaw;
    float3 tint;
    float pad0;
    float3 bmin;
    float pad1;
    float3 bmax;
    float pad2;
};

struct InstanceBufferGPU {
    InstanceGPU* d_instances = nullptr;
    int count = 0;
    int capacity = 0;
};

static void free_instance_buffer(InstanceBufferGPU& b) {
    if (b.d_instances) CUDA_CHECK(cudaFree(b.d_instances));
    b = {};
}

static void ensure_instance_capacity(InstanceBufferGPU& b, int needed) {
    if (needed <= b.capacity) return;
    InstanceGPU* d_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_new, sizeof(InstanceGPU) * needed));
    if (b.d_instances) CUDA_CHECK(cudaFree(b.d_instances));
    b.d_instances = d_new;
    b.capacity = needed;
}

static void upload_instances(InstanceBufferGPU& b, const vector<InstanceGPU>& host) {
    ensure_instance_capacity(b, (int)host.size());
    b.count = (int)host.size();
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(b.d_instances, host.data(), sizeof(InstanceGPU) * host.size(), cudaMemcpyHostToDevice));
    }
}

static float4* d_accum = nullptr;
static uchar3* d_out = nullptr;
static uint64_t* d_seed = nullptr;
static int gW=0, gH=0;

static void init_renderer(int W, int H){
    gW=W; gH=H;
    CUDA_CHECK(cudaMalloc(&d_accum, sizeof(float4)*W*H));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(uchar3)*W*H));
    CUDA_CHECK(cudaMalloc(&d_seed, sizeof(uint64_t)*W*H));
    CUDA_CHECK(cudaMemset(d_accum, 0, sizeof(float4)*W*H));
    std::vector<uint64_t> seeds(W*H);
    uint64_t base = 1469598103934665603ULL;
    for (int i=0;i<W*H;i++){
        uint64_t s = base ^ (uint64_t)i * 1099511628211ULL;
        s ^= s >> 33; s *= 0xff51afd7ed558ccdULL;
        s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ULL;
        s ^= s >> 33;
        seeds[i] = s ? s : 1ULL;
    }
    CUDA_CHECK(cudaMemcpy(d_seed, seeds.data(), sizeof(uint64_t)*W*H, cudaMemcpyHostToDevice));
}
static void shutdown_renderer(){
    if (d_accum) CUDA_CHECK(cudaFree(d_accum));
    if (d_out) CUDA_CHECK(cudaFree(d_out));
    if (d_seed) CUDA_CHECK(cudaFree(d_seed));
    d_accum=nullptr; d_out=nullptr; d_seed=nullptr; gW=gH=0;
}

struct Ray { float3 o, d; };

__host__ __device__ inline uint64_t xorshift64star(uint64_t& s){
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL;
}
__host__ __device__ inline float rnd01(uint64_t& s){
    uint32_t v = (uint32_t)(xorshift64star(s) >> 40); return (float)v / (float)(1u<<24);
}

__device__ inline bool intersect_aabb_bvh(const Ray& r, const float3& bmin, const float3& bmax, float tmax){
    float t0 = 0.0f, t1 = tmax;
    for (int ax=0; ax<3; ax++){
        float ro = (ax==0)? r.o.x : (ax==1)? r.o.y : r.o.z;
        float rd = (ax==0)? r.d.x : (ax==1)? r.d.y : r.d.z;
        float mn = (ax==0)? bmin.x : (ax==1)? bmin.y : bmin.z;
        float mx = (ax==0)? bmax.x : (ax==1)? bmax.y : bmax.z;
        float inv = 1.0f / (fabsf(rd) > 1e-12f ? rd : (rd>=0?1e-12f:-1e-12f));
        float tNear = (mn - ro) * inv, tFar = (mx - ro) * inv;
        if (tNear > tFar) { float tmp=tNear; tNear=tFar; tFar=tmp; }
        t0 = fmaxf(t0, tNear); t1 = fminf(t1, tFar); if (t0 > t1) return false;
    }
    return true;
}
__device__ inline bool intersect_tri(const Ray& r, const Tri& t, float& outT, float& u, float& v){
    float3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
    float3 p = cross3(r.d, e2);
    float det = dot3(e1, p);
    if (fabsf(det) < 1e-8f) return false;
    float inv = 1.0f / det;
    float3 s = r.o - t.v0;
    u = dot3(s, p) * inv; if (u < 0.0f || u > 1.0f) return false;
    float3 q = cross3(s, e1);
    v = dot3(r.d, q) * inv; if (v < 0.0f || (u+v) > 1.0f) return false;
    float tt = dot3(e2, q) * inv; if (tt <= 1e-4f) return false;
    outT = tt; return true;
}

struct Hit {
    float t;
    float3 p, n, albedo;
    int hit;
    int mat;
};

__device__ inline Hit trace_single_scene(const GpuScene& S, const Ray& r){
    Hit best{}; best.t = 1e30f; best.hit = 0; best.mat = 0;
    if (S.root < 0 || S.triCount <= 0 || S.nodeCount <= 0 || S.d_nodes == nullptr) return best;
    const int STACK_MAX = 64;
    int stack[STACK_MAX];
    int sp = 0;
    stack[sp++] = S.root;
    while (sp > 0) {
        int ni = stack[--sp];
        BVHNode node = S.d_nodes[ni];
        if (!intersect_aabb_bvh(r, node.bmin, node.bmax, best.t)) continue;
        if (node.left < 0 && node.right < 0) {
            for (int i=0;i<node.count;i++) {
                int idx = S.d_triIdx[node.start + i];
                Tri t = S.d_tris[idx];
                float tt,u,v;
                if (intersect_tri(r, t, tt, u, v) && tt < best.t) {
                    float w = 1.0f - u - v;
                    float3 n = norm3(t.n0*w + t.n1*u + t.n2*v);
                    if(dot3(n, r.d) > 0.0f) n = n * -1.0f;
                    best.t = tt;
                    best.p = r.o + r.d * tt;
                    best.n = n;
                    best.albedo = t.albedo;
                    best.hit = 1;
                    best.mat = t.materialType;
                }
            }
        } else {
            if (sp + 2 <= STACK_MAX) { stack[sp++] = node.left; stack[sp++] = node.right; }
        }
    }
    return best;
}

__device__ inline float3 rotateYDev(const float3& p, float ang) {
    float c = cosf(ang), s = sinf(ang);
    return make_f3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

__device__ inline Hit trace_instanced_scene(const GpuScene& meshScene, const InstanceGPU* instances, int instanceCount, const Ray& worldRay) {
    Hit best{}; best.t = 1e30f; best.hit = 0; best.mat = 0;
    for (int i = 0; i < instanceCount; ++i) {
        const InstanceGPU inst = instances[i];
        if (!intersect_aabb_bvh(worldRay, inst.bmin, inst.bmax, best.t)) continue;

        Ray localRay;
        localRay.o = rotateYDev(worldRay.o - inst.pos, -inst.yaw);
        localRay.d = rotateYDev(worldRay.d, -inst.yaw);

        Hit h = trace_single_scene(meshScene, localRay);
        if (!h.hit) continue;

        float3 worldP = rotateYDev(h.p, inst.yaw) + inst.pos;
        float3 worldN = norm3(rotateYDev(h.n, inst.yaw));
        float tWorld = len3(worldP - worldRay.o);
        if (tWorld < best.t) {
            best = h;
            best.t = tWorld;
            best.p = worldP;
            best.n = worldN;
            best.albedo = clamp01(h.albedo * inst.tint);
            if(dot3(best.n, worldRay.d) > 0.0f) best.n = best.n * -1.0f;
        }
    }
    return best;
}

__device__ inline Hit trace_scene(const GpuScene& staticS, const GpuScene& meshScene, const InstanceGPU* instances, int instanceCount, const Ray& r){
    Hit a = trace_single_scene(staticS, r);
    Hit b = trace_instanced_scene(meshScene, instances, instanceCount, r);
    if (a.hit && b.hit) return (a.t <= b.t) ? a : b;
    if (a.hit) return a;
    return b;
}

__device__ inline void build_onb(const float3& n, float3& t, float3& b){
    float3 up = (fabsf(n.z) < 0.999f) ? make_f3(0,0,1) : make_f3(0,1,0);
    t = norm3(cross3(up, n)); b = cross3(n, t);
}
__device__ inline float3 cosine_hemisphere(const float3& n, uint64_t& rng){
    float r1 = rnd01(rng), r2 = rnd01(rng);
    float phi = 2.0f * (float)M_PI * r1;
    float r = sqrtf(r2);
    float x = r * cosf(phi), y = r * sinf(phi), z = sqrtf(fmaxf(0.0f, 1.0f - r2));
    float3 t,b; build_onb(n, t, b);
    return norm3(t*x + b*y + n*z);
}

__device__ inline float3 sky_gradient_day(const float3& dir){
    float t = clamp01(0.5f * (dir.y + 1.0f));
    float3 horizon = make_f3(1.00f, 0.96f, 0.92f);
    float3 zenith  = make_f3(0.52f, 0.72f, 1.00f);
    return horizon * (1.0f - t) + zenith * t;
}
__device__ inline float3 sky_gradient_night(const float3& dir){
    float t = clamp01(0.5f * (dir.y + 1.0f));
    float3 horizon = make_f3(0.06f, 0.08f, 0.12f);
    float3 zenith  = make_f3(0.01f, 0.02f, 0.05f);
    return horizon * (1.0f - t) + zenith * t;
}
__device__ inline bool shadow_ray(const GpuScene& staticS, const GpuScene& meshScene, const InstanceGPU* instances, int instanceCount, const float3& p, const float3& dir){
    Ray r; r.o = p + dir * 1e-3f; r.d = dir;
    Hit h = trace_scene(staticS, meshScene, instances, instanceCount, r);
    return h.hit != 0;
}

struct RenderParams {
    int W, H, maxBounces, spp;
    float3 camPos, camFwd, camRgt, camUp;
    float fovY;
    float orthoScale;
    int projectionMode;
    float3 sunDir;
    float sunIntensity, skyIntensity, exposure, persistAlpha;
    float dayFactor;
    float nightSkyBoost;
    int isNight;
    float nightEmissionMult;
};

__device__ inline Ray make_camera_ray(const RenderParams& P, int x, int y, uint64_t& rng){
    float u = ((x + rnd01(rng)) / (float)P.W) * 2.0f - 1.0f;
    float v = ((y + rnd01(rng)) / (float)P.H) * 2.0f - 1.0f;
    float aspect = (float)P.W / (float)P.H;
    Ray r;
    if (P.projectionMode == 0) {
        float3 planeOffset = P.camRgt * (u * aspect * P.orthoScale) + P.camUp * (-v * P.orthoScale);
        r.o = P.camPos + planeOffset;
        r.d = norm3(P.camFwd);
    } else {
        float tanF = tanf(0.5f * P.fovY);
        float3 dir = norm3(P.camFwd + P.camRgt * (u * aspect * tanF) + P.camUp * (-v * tanF));
        r.o = P.camPos;
        r.d = dir;
    }
    return r;
}
__device__ inline float3 environment_light(const float3& dir, const RenderParams& P) {
    float3 skyDay = sky_gradient_day(dir);
    float3 skyNight = sky_gradient_night(dir) * P.nightSkyBoost;
    float3 sky = (skyNight * (1.0f - P.dayFactor) + skyDay * P.dayFactor) * P.skyIntensity;
    float sun = powf(fmaxf(0.0f, dot3(dir, P.sunDir)), 950.0f) * P.sunIntensity;
    return sky + make_f3(1.0f, 0.98f, 0.95f) * sun;
}
__device__ inline float3 radiance_for_pixel(const GpuScene& staticS, const GpuScene& meshScene, const InstanceGPU* instances, int instanceCount, const RenderParams& P, int px, int py, uint64_t& rng){
    Ray ray = make_camera_ray(P, px, py, rng);
    float3 L = make_f3(0,0,0), T = make_f3(1,1,1);
    for (int bounce=0; bounce<P.maxBounces; bounce++) {
        Hit h = trace_scene(staticS, meshScene, instances, instanceCount, ray);
        if (!h.hit) { L = L + T * environment_light(ray.d, P); break; }
        if (h.mat == 3 && P.isNight) {
            L = L + T * h.albedo * P.nightEmissionMult;
        }
        float nl = fmaxf(0.0f, dot3(h.n, P.sunDir));
        if (P.sunIntensity > 0.0001f && nl > 0.0f && !shadow_ray(staticS, meshScene, instances, instanceCount, h.p + h.n * 1e-3f, P.sunDir)) {
            float3 sunC = make_f3(1.0f, 0.96f, 0.90f) * P.sunIntensity;
            L = L + T * (h.albedo * (1.0f / (float)M_PI)) * sunC * nl;
        }
        if (h.mat == 1) {
            L = L + T * (h.albedo * 0.10f);
            float3 refl = norm3(ray.d - h.n * 2.0f * dot3(ray.d, h.n));
            ray.o = h.p + h.n * 1e-3f;
            ray.d = norm3(refl * 0.80f + cosine_hemisphere(h.n, rng) * 0.20f);
            T = T * clamp01(h.albedo * make_f3(0.95f, 1.05f, 1.35f));
        } else {
            float3 newDir = cosine_hemisphere(h.n, rng);
            ray.o = h.p + h.n * 1e-3f;
            ray.d = newDir;
            T = T * h.albedo;
        }
        if (bounce >= 2) {
            float p = fmaxf(0.05f, fminf(0.95f, luminance(T)));
            if (rnd01(rng) > p) break;
            T = T / p;
        }
    }
    return L;
}
__global__ void k_render_pt(GpuScene staticS, GpuScene meshScene, InstanceGPU* instances, int instanceCount, RenderParams P, uint64_t frameIndex, float4* accum, uchar3* out, uint64_t* seeds){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.W || y >= P.H) return;
    int idx = y * P.W + x;
    uint64_t rng = seeds[idx] ^ (frameIndex * 0x9e3779b97f4a7c15ULL);
    float3 sum = make_f3(0,0,0);
    for (int s=0; s<P.spp; s++) sum = sum + radiance_for_pixel(staticS, meshScene, instances, instanceCount, P, x, y, rng);
    seeds[idx] = rng;
    float3 c = sum / (float)P.spp;
    float4 a = accum[idx];
    float alpha = P.persistAlpha;
    a.x = a.x * alpha + c.x * (1.0f - alpha);
    a.y = a.y * alpha + c.y * (1.0f - alpha);
    a.z = a.z * alpha + c.z * (1.0f - alpha);
    a.w = 1.0f;
    accum[idx] = a;
    float3 avg = make_f3(a.x, a.y, a.z);
    float3 mapped = make_f3(1.0f,1.0f,1.0f) - make_f3(expf(-avg.x*P.exposure), expf(-avg.y*P.exposure), expf(-avg.z*P.exposure));
    mapped = clamp01(mapped);
    out[idx] = make_uchar3((unsigned char)(mapped.x*255.0f), (unsigned char)(mapped.y*255.0f), (unsigned char)(mapped.z*255.0f));
}

bool isArrowLeft(int key)  { return key == 81 || key == 2424832 || key == 65361; }
bool isArrowUp(int key)    { return key == 82 || key == 2490368 || key == 65362; }
bool isArrowRight(int key) { return key == 83 || key == 2555904 || key == 65363; }
bool isArrowDown(int key)  { return key == 84 || key == 2621440 || key == 65364; }

struct SunState {
    float3 dir;
    float sunIntensity;
    float skyIntensity;
    float dayFactor;
    float nightSkyBoost;
    bool aboveHorizon;
    float azimuthDeg;
    float altitudeDeg;
};

static SunState computeSunState(time_t curTime) {
    SunState s{};
    std::tm tmv = localTmFromTimeT(curTime);
    double tod = tmv.tm_hour * 3600.0 + tmv.tm_min * 60.0 + tmv.tm_sec;
    double hour = tod / 3600.0;

    const double sunrise = 9.0;
    const double sunset  = 21.0;

    double t = 0.0;
    bool day = false;
    if(hour >= sunrise && hour <= sunset) {
        day = true;
        t = (hour - sunrise) / (sunset - sunrise);
    }

    double azimuthDegBase = -90.0 + t * 180.0;
    double azimuthDeg = azimuthDegBase + (double)SUN_ROTATION_OFFSET_DEG;
    double altitudeDeg = day ? sin(t * M_PI) * 58.0 : -8.0;

    double az  = azimuthDeg * M_PI / 180.0;
    double alt = altitudeDeg * M_PI / 180.0;

    float x = (float)cos(alt) * (float)cos(az);
    float y = (float)sin(alt);
    float z = (float)cos(alt) * (float)sin(az);

    s.dir = norm3(make_f3(x, y, z));
    s.aboveHorizon = day;
    s.azimuthDeg = (float)azimuthDeg;
    s.altitudeDeg = (float)altitudeDeg;

    if(day) {
        float daylight = (float)sin(t * M_PI);
        daylight = clamp01(daylight);
        s.dayFactor = daylight;
        s.sunIntensity = 0.35f + daylight * 3.65f;
        s.skyIntensity = 0.22f + daylight * 1.08f;
        s.nightSkyBoost = 1.0f;
    } else {
        s.dayFactor = 0.0f;
        s.sunIntensity = 0.0f;
        s.skyIntensity = 0.18f;
        s.nightSkyBoost = 1.35f;
    }

    return s;
}

static float computeFrameAverageLuminance(const Mat& frameBGR) {
    if(frameBGR.empty()) return 0.0f;
    const int total = frameBGR.rows * frameBGR.cols;
    if(total <= 0) return 0.0f;
    double sum = 0.0;
    for(int y = 0; y < frameBGR.rows; ++y) {
        const Vec3b* row = frameBGR.ptr<Vec3b>(y);
        for(int x = 0; x < frameBGR.cols; ++x) {
            float b = row[x][0] / 255.0f;
            float g = row[x][1] / 255.0f;
            float r = row[x][2] / 255.0f;
            sum += 0.0722 * b + 0.7152 * g + 0.2126 * r;
        }
    }
    return (float)(sum / (double)total);
}

static AABB worldAABBForInstance(const ObjInstanceTemplate& T, const float3& pos, float yaw) {
    const float3 mn = T.localBoundsNormalized.bmin;
    const float3 mx = T.localBoundsNormalized.bmax;
    float3 corners[8] = {
        make_f3(mn.x,mn.y,mn.z), make_f3(mx.x,mn.y,mn.z), make_f3(mn.x,mx.y,mn.z), make_f3(mx.x,mx.y,mn.z),
        make_f3(mn.x,mn.y,mx.z), make_f3(mx.x,mn.y,mx.z), make_f3(mn.x,mx.y,mx.z), make_f3(mx.x,mx.y,mx.z)
    };
    AABB out = aabb_empty();
    for (int i=0;i<8;i++) aabb_extend(out, rotateY(corners[i], yaw) + pos);
    const float eps = 0.02f;
    out.bmin = out.bmin - make_f3(eps, eps, eps);
    out.bmax = out.bmax + make_f3(eps, eps, eps);
    return out;
}


static bool applyControlCommand(const string& rawCmd, Camera2D& cam, Player& player, const vector<vector<Terrain>>& terrain, vector<Persona>& agents, bool& autoExposure, float& exposure, long long currentRealMs, long long& lastPlayerInputMs) {
    string cmd = upperStr(trimStr(rawCmd));
    if(cmd.empty()) return false;

    auto oneFloat = [&](const string& prefix, float& out)->bool {
        if(!startsWith(cmd, prefix)) return false;
        string rest = trimStr(cmd.substr(prefix.size()));
        if(rest.empty()) return false;
        try { out = stof(rest); return true; } catch(...) { return false; }
    };

    bool movedPlayer = false;
    if(cmd == "ZOOM_IN") cam.changeZoom(CAMERA_ZOOM_STEP);
    else if(cmd == "ZOOM_OUT") cam.changeZoom(-CAMERA_ZOOM_STEP);
    else if(cmd == "ORBIT_LEFT") { cam.followPlayer = false; cam.conicPerspective = true; cam.changeOrbitDeg(-CAMERA_ORBIT_STEP_DEG * 3.0f); }
    else if(cmd == "ORBIT_RIGHT") { cam.followPlayer = false; cam.conicPerspective = true; cam.changeOrbitDeg(CAMERA_ORBIT_STEP_DEG * 3.0f); }
    else if(cmd == "TILT_UP") { cam.followPlayer = false; cam.conicPerspective = true; cam.changeTilt(CAMERA_TILT_STEP * 2.0f); }
    else if(cmd == "TILT_DOWN") { cam.followPlayer = false; cam.conicPerspective = true; cam.changeTilt(-CAMERA_TILT_STEP * 2.0f); }
    else if(cmd == "NEXT_NPC") cam.nextNpcFocus(agents);
    else if(cmd == "PREV_NPC") cam.prevNpcFocus(agents);
    else if(cmd == "FOLLOW_PLAYER") cam.followPlayer = true;
    else if(cmd == "FOLLOW_NPC") cam.followPlayer = false;
    else if(cmd == "TOGGLE_FOLLOW") cam.followPlayer = !cam.followPlayer;
    else if(cmd == "MOVE_UP") movedPlayer = player.tryMove(-1,0,terrain);
    else if(cmd == "MOVE_DOWN") movedPlayer = player.tryMove(1,0,terrain);
    else if(cmd == "MOVE_LEFT") movedPlayer = player.tryMove(0,-1,terrain);
    else if(cmd == "MOVE_RIGHT") movedPlayer = player.tryMove(0,1,terrain);
    else if(cmd == "AUTOEXPOSURE_ON") autoExposure = true;
    else if(cmd == "AUTOEXPOSURE_OFF") autoExposure = false;
    else if(cmd == "TOGGLE_AUTOEXPOSURE") autoExposure = !autoExposure;
    else if(cmd == "PROJECTION_CONIC") cam.conicPerspective = true;
    else if(cmd == "PROJECTION_ORTHO") cam.conicPerspective = false;
    else if(cmd == "TOGGLE_PROJECTION") cam.toggleProjectionMode();
    else if(cmd == "FOV_INC") cam.changePerspectiveFov(PERSPECTIVE_FOV_STEP_DEG);
    else if(cmd == "FOV_DEC") cam.changePerspectiveFov(-PERSPECTIVE_FOV_STEP_DEG);
    else if(cmd == "EXPOSURE_INC") { exposure = min(20.0f, exposure * 1.10f); autoExposure = false; }
    else if(cmd == "EXPOSURE_DEC") { exposure = max(0.05f, exposure * 0.90f); autoExposure = false; }
    else if(cmd == "EXPOSURE_RESET") { exposure = 1.25f; autoExposure = true; }
    else if(cmd == "AUTO_RETURN_ON") cam.autoReturnToPlayerWhenMoving = true;
    else if(cmd == "AUTO_RETURN_OFF") cam.autoReturnToPlayerWhenMoving = false;
    else {
        float v = 0.0f;
        if(oneFloat("SET_ZOOM ", v)) cam.setZoom(v);
        else if(oneFloat("SET_TILT ", v)) { cam.followPlayer = false; cam.conicPerspective = true; cam.setTilt(v); }
        else if(oneFloat("SET_ORBIT_DEG ", v)) { cam.followPlayer = false; cam.conicPerspective = true; cam.setOrbitDeg(v); }
        else if(oneFloat("SET_FOV ", v)) cam.perspectiveFovDeg = clampT(v, PERSPECTIVE_FOV_MIN_DEG, PERSPECTIVE_FOV_MAX_DEG);
        else if(oneFloat("SET_EXPOSURE ", v)) { exposure = clampT(v, 0.05f, 20.0f); autoExposure = false; }
        else return false;
    }

    if(movedPlayer) {
        lastPlayerInputMs = currentRealMs;
        if(cam.autoReturnToPlayerWhenMoving) cam.followPlayer = true;
    }
    return true;
}

int main() {
    Mat img = imread("casas.png");
    if(img.empty()) { printf("Failed to load casas.png\n"); return -1; }
    Mat zimg = imread("z.png");
    if(zimg.empty()) { printf("Failed to load z.png\n"); return -1; }
    if(img.size() != zimg.size()) { printf("casas.png and z.png must have exactly the same resolution\n"); return -1; }

    vector<Tri> baseObjTris;
    if (!load_obj_with_mtl("base.obj", baseObjTris)) {
        printf("Failed to load base.obj / base.mtl from current folder\n");
        return -1;
    }
    ObjInstanceTemplate objTemplate = prepareObjTemplate(baseObjTris);
    printf("Loaded OBJ triangles: %d | instance scale: %.4f\n", (int)baseObjTris.size(), objTemplate.scale);

    auto terrain = buildTerrain(img);
    auto zmap = buildZOffsets(zimg, Z_HEIGHT_MULTIPLIER);
    int H = (int)terrain.size(), W = (int)terrain[0].size();

    vector<IPoint> walkables, bedrooms, works, kitchens, wcs, livings, parks, seawaters, schools, highschools, universities;
    for(int y = 0; y < H; ++y) for(int x = 0; x < W; ++x) {
        Terrain t = terrain[y][x];
        if(isWalkable(t)) walkables.emplace_back(y, x);
        if(t == BEDROOM) bedrooms.emplace_back(y, x);
        if(t == WORK) works.emplace_back(y, x);
        if(t == KITCHEN) kitchens.emplace_back(y, x);
        if(t == WC) wcs.emplace_back(y, x);
        if(t == LIVING) livings.emplace_back(y, x);
        if(t == PARK) parks.emplace_back(y, x);
        if(t == SEAWATER) seawaters.emplace_back(y, x);
        if(t == SCHOOL) schools.emplace_back(y, x);
        if(t == HIGHSCHOOL) highschools.emplace_back(y, x);
        if(t == UNIVERSITY) universities.emplace_back(y, x);
    }

    if(walkables.empty() || bedrooms.empty() || kitchens.empty() || wcs.empty() || livings.empty()) {
        printf("Map is missing one or more categories: bedroom/kitchen/wc/living.\n");
        return -1;
    }
    if((int)bedrooms.size() < numAgents + 1) {
        printf("Not enough bedroom cells for agents + player. Bedrooms available: %d\n", (int)bedrooms.size());
        return -1;
    }

    vector<District> districts;
    auto districtMap = buildDistrictMap(terrain, districts);

    vector<IPoint> availableBedrooms = bedrooms;
    shuffle(availableBedrooms.begin(), availableBedrooms.end(), rng);
    vector<IPoint> availableWorks = works;
    shuffle(availableWorks.begin(), availableWorks.end(), rng);

    Player player;
    IPoint playerStart = availableBedrooms.back();
    availableBedrooms.pop_back();
    player.init(playerStart);

    vector<Persona> agents;
    agents.reserve(numAgents);

    for(int i = 0; i < numAgents; ++i) {
        IPoint bedroom = availableBedrooms.back();
        availableBedrooms.pop_back();
        int hd = districtMap[bedroom.first][bedroom.second];
        vector<IPoint> localKitchens = districts[hd].kitchens.empty() ? kitchens : districts[hd].kitchens;
        vector<IPoint> localLivings  = districts[hd].livings.empty()  ? livings  : districts[hd].livings;
        vector<IPoint> localParks    = districts[hd].parks.empty()    ? parks    : districts[hd].parks;
        vector<IPoint> localSea      = districts[hd].seawaters.empty() ? seawaters : districts[hd].seawaters;
        if(localParks.empty()) localParks = districts[hd].walkPaths;
        if(localParks.empty()) localParks = districts[hd].asphalt;
        if(localParks.empty()) localParks = walkables;

        Persona a;
        a.init(i, bedroom);
        a.age = irand(1, 90);
        a.homeDistrict = hd;
        a.bedroom = bedroom;
        a.localKitchens = localKitchens;
        a.localLivings = localLivings;
        a.localWCs = wcs;
        a.localParks = localParks;
        a.localSea = localSea.empty() ? seawaters : localSea;
        a.hunger = 10.0 + frand01() * 30.0;
        a.bladder = frand01() * 20.0;
        a.energy = 65.0 + frand01() * 25.0;
        a.social = 45.0 + frand01() * 35.0;

        DayRole role = a.getRole();
        if(role == ROLE_ADULT_WORK) {
            if(!availableWorks.empty()) { a.workplace = availableWorks.back(); availableWorks.pop_back(); }
            else if(!works.empty()) a.workplace = randomFromList(works);
            else a.workplace = bedroom;
            a.studyplace = a.workplace;
        } else if(role == ROLE_CHILD_SCHOOL) {
            a.studyplace = !schools.empty() ? randomFromList(schools) : bedroom;
            a.workplace = a.studyplace;
        } else if(role == ROLE_TEEN_HIGHSCHOOL) {
            a.studyplace = !highschools.empty() ? randomFromList(highschools) : bedroom;
            a.workplace = a.studyplace;
        } else if(role == ROLE_YOUNG_UNIVERSITY) {
            a.studyplace = !universities.empty() ? randomFromList(universities) : bedroom;
            a.workplace = a.studyplace;
        } else {
            a.studyplace = !a.localSea.empty() ? randomFromList(a.localSea) : bedroom;
            a.workplace = a.studyplace;
        }
        agents.push_back(a);
    }

    Camera2D cam;
    cam.centerX = player.renderCell().x;
    cam.centerY = player.renderCell().y;
    cam.zoom = CAMERA_NPC_ZOOM_FACTOR;
    cam.targetCenterX = cam.centerX;
    cam.targetCenterY = cam.centerY;
    cam.targetZoom = cam.zoom;
    cam.manualZoom = CAMERA_NPC_ZOOM_FACTOR;
    cam.focusedNpc = 0;
    cam.followPlayer = true;

    vector<Tri> staticTris;
    buildStaticTerrainMesh(terrain, zmap, staticTris);

    GpuScene staticScene{};
    GpuScene npcMeshScene{};
    if(!upload_scene(staticTris, staticScene)) {
        printf("Failed to upload static scene\n");
        return -1;
    }
    if(!upload_scene(objTemplate.tris, npcMeshScene)) {
        printf("Failed to upload shared NPC mesh scene\n");
        free_scene(staticScene);
        return -1;
    }

    init_renderer(SCREEN_W, SCREEN_H);

    namedWindow("SimPT Option B Solar OBJ Instanced", WINDOW_NORMAL);
    resizeWindow("SimPT Option B Solar OBJ Instanced", SCREEN_W, SCREEN_H);

    float exposure = 1.25f;
    bool autoExposure = true;
    time_t startTimestamp = makeStartTimestamp(START_YEAR, START_MONTH, START_DAY, START_HOUR, START_MINUTE, START_SECOND);
    time_t curTime = startTimestamp;
    double simTickAccumulator = 0.0;
    long long lastPlayerInputMs = nowMs();
    long long prevRealMs = lastPlayerInputMs;
    uint64_t frameIndex = 0;
    vector<Vec3b> hostFrame(SCREEN_W * SCREEN_H);

    UdpReceiver udp;
    bool udpOk = udp.init(UDP_CONTROL_PORT);
    cout << "UDP control: " << (udpOk ? "listening on port " + to_string(UDP_CONTROL_PORT) : "disabled") << "\n";

    InstanceBufferGPU instanceBuffer{};
    vector<InstanceGPU> hostInstances;
    hostInstances.reserve(agents.size() + 1);

    VideoWriter timelapseWriter;
    string timelapseFilename;
    long long runEpoch = (long long)time(nullptr);
    long long lastSavedMinuteBucket = -1;
    if (WRITE_TIMELAPSE_VIDEO) {
        timelapseFilename = string("simulation_timelapse_") + to_string(runEpoch) + string(".mp4");
        int fourcc = VideoWriter::fourcc('m','p','4','v');
        if (!timelapseWriter.open(timelapseFilename, fourcc, VIDEO_FPS, Size(SCREEN_W, SCREEN_H), true)) {
            cerr << "Warning: could not open video writer for " << timelapseFilename << "\n";
        } else {
            cout << "Writing timelapse to: " << timelapseFilename << "\n";
        }
    }

    while(true) {
        long long currentRealMs = nowMs();
        double deltaRealSec = max(0.0, (double)(currentRealMs - prevRealMs) / 1000.0);
        prevRealMs = currentRealMs;

        simTickAccumulator += deltaRealSec * SIM_TICKS_PER_REAL_SECOND;

        for(const string& udpCmd : udp.recvAll()) {
            applyControlCommand(udpCmd, cam, player, terrain, agents, autoExposure, exposure, currentRealMs, lastPlayerInputMs);
        }

        int key = waitKeyEx(1);
        if(key == 27) break;

        if(key == '+' || key == '=') cam.changeZoom(CAMERA_ZOOM_STEP);
        if(key == '-' || key == '_') cam.changeZoom(-CAMERA_ZOOM_STEP);
        if(isArrowLeft(key))  { cam.followPlayer = false; cam.conicPerspective = true; cam.changeOrbitDeg(-CAMERA_ORBIT_STEP_DEG * 3.0f); }
        if(isArrowRight(key)) { cam.followPlayer = false; cam.conicPerspective = true; cam.changeOrbitDeg( CAMERA_ORBIT_STEP_DEG * 3.0f); }
        if(isArrowUp(key))    { cam.followPlayer = false; cam.conicPerspective = true; cam.changeTilt( CAMERA_TILT_STEP * 2.0f); }
        if(isArrowDown(key))  { cam.followPlayer = false; cam.conicPerspective = true; cam.changeTilt(-CAMERA_TILT_STEP * 2.0f); }
        if(key == 32) { cam.nextNpcFocus(agents); }
        if(key == 'f' || key == 'F') { cam.followPlayer = !cam.followPlayer; }

        bool movedPlayer = false;
        if(key == 'w' || key == 'W') movedPlayer = player.tryMove(-1, 0, terrain);
        if(key == 's' || key == 'S') movedPlayer = player.tryMove( 1, 0, terrain);
        if(key == 'a' || key == 'A') movedPlayer = player.tryMove( 0,-1, terrain);
        if(key == 'd' || key == 'D') movedPlayer = player.tryMove( 0, 1, terrain);

        if(key == 'e' || key == 'E') autoExposure = !autoExposure;
        if(key == 'p' || key == 'P') cam.toggleProjectionMode();
        if(key == ',') cam.changePerspectiveFov(-PERSPECTIVE_FOV_STEP_DEG);
        if(key == '.') cam.changePerspectiveFov( PERSPECTIVE_FOV_STEP_DEG);
        if(key == '[') { exposure = max(0.05f, exposure * 0.90f); autoExposure = false; }
        if(key == ']') { exposure = min(20.0f, exposure * 1.10f); autoExposure = false; }
        if(key == '0') { exposure = 1.25f; autoExposure = true; }

        if(movedPlayer) {
            lastPlayerInputMs = currentRealMs;
            if(cam.autoReturnToPlayerWhenMoving) cam.followPlayer = true;
        }

        int simSteps = 0;
        while(simTickAccumulator >= 1.0 && simSteps < MAX_SIM_STEPS_PER_FRAME) {
            curTime += (time_t)SIM_TICK_SECONDS;
            for(auto &a : agents) a.stepDecision(terrain, curTime, SIM_TICK_SECONDS);
            simTickAccumulator -= 1.0;
            simSteps++;
        }
        if(simSteps == MAX_SIM_STEPS_PER_FRAME && simTickAccumulator > 10.0) simTickAccumulator = 10.0;

        for(auto &a : agents) a.updateMovementOnly();
        player.update();
        cam.updatePlayerOrCinematic(player, agents, currentRealMs, lastPlayerInputMs);

        Camera3D cam3 = buildCamera3D(cam, zmap);

        hostInstances.clear();
        for(const auto& a : agents) {
            float3 p = cellToWorld(a.renderCell().y, a.renderCell().x, zmap);
            float3 worldPos = make_f3(p.x + TILE_WORLD * 0.5f, p.y + FLOOR_THICKNESS_3D, p.z + TILE_WORLD * 0.5f);
            float yaw = -a.facingAngle + (float)CV_PI * 0.5f;
            float3 tint = make_f3(
                0.55f + 0.10f * ((a.colorVariant + 0) % 3),
                0.55f + 0.12f * ((a.colorVariant + 1) % 3),
                0.60f + 0.08f * ((a.colorVariant + 2) % 3)
            );
            AABB bb = worldAABBForInstance(objTemplate, worldPos, yaw);
            InstanceGPU inst{};
            inst.pos = worldPos;
            inst.yaw = yaw;
            inst.tint = tint;
            inst.bmin = bb.bmin;
            inst.bmax = bb.bmax;
            hostInstances.push_back(inst);
        }
        {
            float3 p = cellToWorld(player.renderCell().y, player.renderCell().x, zmap);
            float3 worldPos = make_f3(p.x + TILE_WORLD * 0.5f, p.y + FLOOR_THICKNESS_3D, p.z + TILE_WORLD * 0.5f);
            float yaw = -player.facingAngle + (float)CV_PI * 0.5f;
            AABB bb = worldAABBForInstance(objTemplate, worldPos, yaw);
            InstanceGPU inst{};
            inst.pos = worldPos;
            inst.yaw = yaw;
            inst.tint = make_f3(1.10f, 0.42f, 0.35f);
            inst.bmin = bb.bmin;
            inst.bmax = bb.bmax;
            hostInstances.push_back(inst);
        }
        upload_instances(instanceBuffer, hostInstances);

        float3 fwd = norm3(cam3.target - cam3.pos);
        float3 rgt = norm3(cross3(fwd, cam3.up));
        float3 up  = norm3(cross3(rgt, fwd));

        SunState sun = computeSunState(curTime);

        RenderParams P{};
        P.W = SCREEN_W;
        P.H = SCREEN_H;
        P.maxBounces = 4;
        P.spp = 1;
        P.camPos = cam3.pos;
        P.camFwd = fwd;
        P.camRgt = rgt;
        P.camUp  = up;
        P.fovY = cam3.fovY * (float)M_PI / 180.0f;
        P.orthoScale = cam3.orthoScale;
        P.projectionMode = cam3.projectionMode;
        P.sunDir = sun.dir;
        P.sunIntensity = sun.sunIntensity;
        P.skyIntensity = sun.skyIntensity;
        P.exposure = exposure;
        P.persistAlpha = 0.80f;
        P.dayFactor = sun.dayFactor;
        P.nightSkyBoost = sun.nightSkyBoost;
        int hour = getHour(curTime);
        bool isNight = (hour >= 21 || hour < 9);
        P.isNight = isNight ? 1 : 0;
        P.nightEmissionMult = NIGHT_EMISSION_MULT;

        dim3 block(16,16);
        dim3 grid((SCREEN_W + block.x - 1)/block.x, (SCREEN_H + block.y - 1)/block.y);
        k_render_pt<<<grid, block>>>(staticScene, npcMeshScene, instanceBuffer.d_instances, instanceBuffer.count, P, frameIndex, d_accum, d_out, d_seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(hostFrame.data(), d_out, sizeof(uchar3)*SCREEN_W*SCREEN_H, cudaMemcpyDeviceToHost));
        Mat frame(SCREEN_H, SCREEN_W, CV_8UC3, hostFrame.data());

        if(autoExposure) {
            float avgLuma = computeFrameAverageLuminance(frame);
            float safeLuma = max(0.001f, avgLuma);
            float targetExposure = exposure * (AUTOEXPOSURE_TARGET_LUMA / safeLuma);
            targetExposure = clampT(targetExposure, AUTOEXPOSURE_MIN, AUTOEXPOSURE_MAX);
            exposure += (targetExposure - exposure) * AUTOEXPOSURE_ADAPT;
            exposure = clampT(exposure, AUTOEXPOSURE_MIN, AUTOEXPOSURE_MAX);
        }

        Mat shown = frame.clone();

        const int hudX = 20;
        const int hudY = 20;
        const int hudW = min(920, SCREEN_W - 40);
        const int hudH = 180;

        rectangle(shown, Rect(hudX, hudY, hudW, hudH), Scalar(0,0,0), FILLED);
        rectangle(shown, Rect(hudX, hudY, hudW, hudH), Scalar(95,95,95), 1);

        putText(shown, secondsToDateTimeStr(curTime), Point(hudX + 15, hudY + 30), FONT_HERSHEY_SIMPLEX, 0.72, Scalar(255,255,255), 2, LINE_AA);

        string priorityText;
        if(!agents.empty()) {
            int npcIndex = clampT(cam.focusedNpc, 0, (int)agents.size() - 1);
            const Persona& npc = agents[npcIndex];
            priorityText = "NPC " + to_string(npcIndex) + " prioridad: " + npc.currentGoalLabel;
            if(cam.followPlayer) priorityText += " (camara en player)";
        } else {
            priorityText = "Sin NPCs";
        }
        putText(shown, priorityText, Point(hudX + 15, hudY + 58), FONT_HERSHEY_SIMPLEX, 0.62, Scalar(220,220,220), 1, LINE_AA);

        string perfText = "Instancing activo | shared mesh tris=" + to_string((int)baseObjTris.size()) +
                          " | instances=" + to_string(instanceBuffer.count) +
                          " | static tris=" + to_string((int)staticTris.size()) +
                          " | trees on park=" + string(ENABLE_TREES_ON_PARK ? "yes" : "no");
        putText(shown, perfText, Point(hudX + 15, hudY + 84), FONT_HERSHEY_SIMPLEX, 0.54, Scalar(200,200,200), 1, LINE_AA);

        string exposureMode = string("Exposure=") + to_string(exposure).substr(0,5) + (autoExposure ? " | auto" : " | manual");
        putText(shown, exposureMode, Point(hudX + 15, hudY + 108), FONT_HERSHEY_SIMPLEX, 0.56, Scalar(200,200,200), 1, LINE_AA);

        string projText = string("Projection=") + (cam.conicPerspective ? "conic" : "orthographic") +
                          " | FOV=" + to_string(cam.perspectiveFovDeg).substr(0,5) +
                          " | zoom=" + to_string(cam.zoom).substr(0,5);
        putText(shown, projText, Point(hudX + 15, hudY + 132), FONT_HERSHEY_SIMPLEX, 0.56, Scalar(200,200,200), 1, LINE_AA);

        string helpText = "UDP port " + to_string(UDP_CONTROL_PORT) + " | E autoexp | [ ] manual | 0 reset | P projection | , . FOV";
        putText(shown, helpText, Point(hudX + 15, hudY + 156), FONT_HERSHEY_SIMPLEX, 0.50, Scalar(185,185,185), 1, LINE_AA);

        if (WRITE_TIMELAPSE_VIDEO && timelapseWriter.isOpened()) {
            long long minuteBucket = (long long)(curTime / (VIDEO_SIM_MINUTES_PER_FRAME * 60));
            if (minuteBucket != lastSavedMinuteBucket) {
                timelapseWriter.write(shown);
                lastSavedMinuteBucket = minuteBucket;
            }
        }

        imshow("SimPT Option B Solar OBJ Instanced", shown);
        frameIndex++;
    }

    udp.shutdown();

    if (timelapseWriter.isOpened()) {
        timelapseWriter.release();
        cout << "Timelapse saved to: " << timelapseFilename << "\n";
    }

    free_instance_buffer(instanceBuffer);
    shutdown_renderer();
    free_scene(npcMeshScene);
    free_scene(staticScene);
    return 0;
}
