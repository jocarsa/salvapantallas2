#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>

using namespace cv;
using namespace std;

// ============================================================
// SCREEN / WORLD
// ============================================================
const int SCREEN_W = 1920;
const int SCREEN_H = 1080;

const int MAP_W_EXPECTED = 120;
const int MAP_H_EXPECTED = 120;

const int numAgents = 28;
const int timeStepSeconds = 20;

const int TILE_W = 24;
const int TILE_H = 12;
const int FLOOR_THICKNESS = 6;

// Muros
const int WALL_HEIGHT = 12;        // altura visible del muro
const int WALL_HEIGHT_OFFSET = 6; // elevacion del muro por encima del suelo

// ============================================================
// CAMERA TUNING
// ============================================================
const float CAMERA_NPC_ZOOM_FACTOR = 2.10f;
const float CAMERA_CENTER_SMOOTH = 0.045f;
const float CAMERA_ZOOM_SMOOTH   = 0.045f;
const float CAMERA_ANGLE_SMOOTH  = 0.085f;
const long long PLAYER_IDLE_FALLBACK_MS = 5000;

const float CAMERA_ZOOM_MIN = 0.60f;
const float CAMERA_ZOOM_MAX = 4.50f;
const float CAMERA_ZOOM_STEP = 0.18f;

const float CAMERA_ORBIT_STEP_DEG = 7.5f;
const float CAMERA_TILT_STEP = 0.08f;
const float CAMERA_TILT_MIN = 0.45f;
const float CAMERA_TILT_MAX = 1.65f;

// ============================================================
// Z HEIGHT MAP
// ============================================================
// 127 = neutral
// >127 = up
// <127 = down
const float Z_HEIGHT_MULTIPLIER = 4.32f;

// ============================================================
// TERRAIN
// ============================================================
enum Terrain {
    VOID_TERRAIN = 0,
    ASPHALT,      // ff00ff
    WALKABLE,     // 00ff00
    WALL,         // ffffff
    PARK,         // 00c800
    WORK,         // 7f7f7f
    WC,           // ff0000
    LIVING,       // 00ffff
    BEDROOM,      // 0000ff
    KITCHEN,      // ffff00
    SEAWATER,     // 4c48c2
    SCHOOL,       // d58432
    HIGHSCHOOL,   // 8e5f30
    UNIVERSITY    // 70553b
};

typedef pair<int,int> IPoint;

// ============================================================
// GLOBALS
// ============================================================
static unordered_map<string, vector<IPoint>> pathCache;
static mt19937 rng((unsigned int)time(nullptr));

// ============================================================
// UTILS
// ============================================================
string makeKey(const IPoint &start, const IPoint &goal) {
    return to_string(start.first) + "," + to_string(start.second) +
           "->" +
           to_string(goal.first) + "," + to_string(goal.second);
}

inline int manhattan(const IPoint &a, const IPoint &b) {
    return abs(a.first - b.first) + abs(a.second - b.second);
}

double frand01() {
    return uniform_real_distribution<double>(0.0, 1.0)(rng);
}

int irand(int a, int b) {
    return uniform_int_distribution<int>(a, b)(rng);
}

template<typename T>
T clampT(const T &v, const T &lo, const T &hi) {
    return max(lo, min(hi, v));
}

string secondsToDateTimeStr(long seconds) {
    long days = seconds / 86400;
    int year = 2026 + (int)(days / (12 * 31));
    int month = (int)((days / 31) % 12) + 1;
    int day = (int)(days % 31) + 1;
    static const vector<string> wd = {
        "Domingo","Lunes","Martes","Miercoles","Jueves","Viernes","Sabado"
    };
    string dow = wd[days % 7];
    int hr = (seconds % 86400) / 3600;
    int mn = (seconds % 3600) / 60;
    int sc = seconds % 60;

    ostringstream oss;
    oss << setfill('0')
        << year << ":" << setw(2) << month << ":" << setw(2) << day << ":"
        << dow << ":" << setw(2) << hr << ":" << setw(2) << mn << ":" << setw(2) << sc;
    return oss.str();
}

int getHour(long seconds) {
    return (seconds % 86400) / 3600;
}

int getDayIndex(long seconds) {
    return (int)((seconds / 86400) % 7);
}

bool isWeekend(long seconds) {
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

long long nowMs() {
    return (long long)getTickCount() * 1000 / getTickFrequency();
}

// ============================================================
// TERRAIN DECODING
// ============================================================
bool isWalkable(Terrain t) {
    return t != VOID_TERRAIN && t != WALL;
}

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
    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            terrain[y][x] = decodeTerrain(img.at<Vec3b>(y, x));
        }
    }
    return terrain;
}

// ============================================================
// Z MAP
// ============================================================
vector<vector<float>> buildZOffsets(const Mat &img, float multiplier) {
    int h = img.rows, w = img.cols;
    vector<vector<float>> zmap(h, vector<float>(w, 0.0f));

    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            int r = img.at<Vec3b>(y, x)[2];
            zmap[y][x] = ((float)r - 127.0f) * multiplier;
        }
    }
    return zmap;
}

float sampleZNearest(const vector<vector<float>> &zmap, int row, int col) {
    int H = (int)zmap.size();
    int W = (int)zmap[0].size();
    row = clampT(row, 0, H - 1);
    col = clampT(col, 0, W - 1);
    return zmap[row][col];
}

float sampleZBilinear(const vector<vector<float>> &zmap, float row, float col) {
    int H = (int)zmap.size();
    int W = (int)zmap[0].size();

    row = clampT(row, 0.0f, (float)(H - 1));
    col = clampT(col, 0.0f, (float)(W - 1));

    int r0 = (int)floor(row);
    int c0 = (int)floor(col);
    int r1 = min(r0 + 1, H - 1);
    int c1 = min(c0 + 1, W - 1);

    float fr = row - (float)r0;
    float fc = col - (float)c0;

    float z00 = zmap[r0][c0];
    float z01 = zmap[r0][c1];
    float z10 = zmap[r1][c0];
    float z11 = zmap[r1][c1];

    float z0 = z00 * (1.0f - fc) + z01 * fc;
    float z1 = z10 * (1.0f - fc) + z11 * fc;

    return z0 * (1.0f - fr) + z1 * fr;
}

// ============================================================
// PATHFINDING
// ============================================================
vector<IPoint> aStar(const vector<vector<Terrain>> &terrain, const IPoint &start, const IPoint &goal) {
    if(start == goal) return {};

    int H = (int)terrain.size();
    int W = (int)terrain[0].size();

    auto keyStr = [&](const IPoint &pt) {
        return to_string(pt.first) + "," + to_string(pt.second);
    };

    unordered_map<string, IPoint> cameFrom;
    unordered_map<string, int> gScore, fScore;

    auto cmp = [&](const IPoint &a, const IPoint &b) {
        return fScore[keyStr(a)] > fScore[keyStr(b)];
    };

    priority_queue<IPoint, vector<IPoint>, decltype(cmp)> open(cmp);

    gScore[keyStr(start)] = 0;
    fScore[keyStr(start)] = manhattan(start, goal);
    open.push(start);

    vector<IPoint> dirs = {{1,0},{-1,0},{0,1},{0,-1}};

    while(!open.empty()) {
        IPoint current = open.top();
        open.pop();

        if(current == goal) {
            vector<IPoint> path;
            IPoint cur = current;
            while(cameFrom.count(keyStr(cur))) {
                path.push_back(cur);
                cur = cameFrom[keyStr(cur)];
            }
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

// ============================================================
// REGIONS / FACILITIES
// ============================================================
struct District {
    int id = -1;
    vector<IPoint> walkables;
    vector<IPoint> asphalt;
    vector<IPoint> walkPaths;
    vector<IPoint> parks;
    vector<IPoint> works;
    vector<IPoint> wcs;
    vector<IPoint> livings;
    vector<IPoint> bedrooms;
    vector<IPoint> kitchens;
    vector<IPoint> seawaters;
    vector<IPoint> schools;
    vector<IPoint> highschools;
    vector<IPoint> universities;
};

vector<vector<int>> buildDistrictMap(const vector<vector<Terrain>> &terrain, vector<District> &districts) {
    int H = (int)terrain.size();
    int W = (int)terrain[0].size();

    vector<vector<int>> comp(H, vector<int>(W, -1));
    vector<IPoint> dirs = {{1,0},{-1,0},{0,1},{0,-1}};
    int nextId = 0;

    for(int y = 0; y < H; ++y) {
        for(int x = 0; x < W; ++x) {
            if(comp[y][x] != -1) continue;
            if(!isWalkable(terrain[y][x])) continue;

            queue<IPoint> q;
            q.push({y,x});
            comp[y][x] = nextId;

            District d;
            d.id = nextId;

            while(!q.empty()) {
                IPoint p = q.front();
                q.pop();

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
                    int ny = r + dd.first;
                    int nx = c + dd.second;
                    if(ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
                    if(comp[ny][nx] != -1) continue;
                    if(!isWalkable(terrain[ny][nx])) continue;
                    comp[ny][nx] = nextId;
                    q.push({ny,nx});
                }
            }

            districts.push_back(d);
            nextId++;
        }
    }

    return comp;
}

IPoint nearestFromList(const IPoint &from, const vector<IPoint> &list) {
    if(list.empty()) return from;
    return *min_element(list.begin(), list.end(), [&](const IPoint &a, const IPoint &b) {
        return manhattan(from, a) < manhattan(from, b);
    });
}

IPoint randomFromList(const vector<IPoint> &list) {
    if(list.empty()) return {0,0};
    return list[irand(0, (int)list.size() - 1)];
}

IPoint nearestNearHome(const IPoint &homeCell, const vector<IPoint> &list) {
    return nearestFromList(homeCell, list);
}

IPoint nearestBathroomFromCurrent(const IPoint &currentCell, const vector<IPoint> &list) {
    return nearestFromList(currentCell, list);
}

// ============================================================
// SMOOTH MOVER
// ============================================================
struct SmoothMover {
    IPoint cell = {0,0};
    IPoint moveFromCell = {0,0};
    IPoint moveToCell = {0,0};

    Point2f renderCell = Point2f(0,0);
    float moveProgress = 1.0f;
    float moveSpeed = 0.22f;

    bool isMoving() const {
        return moveProgress < 1.0f;
    }

    void initAt(const IPoint &start, float spd) {
        cell = start;
        moveFromCell = start;
        moveToCell = start;
        renderCell = Point2f((float)start.second, (float)start.first);
        moveProgress = 1.0f;
        moveSpeed = spd;
    }

    bool canStartNewStep() const {
        return moveProgress >= 1.0f;
    }

    void startMoveTo(const IPoint &next) {
        moveFromCell = cell;
        moveToCell = next;
        moveProgress = 0.0f;
    }

    void update() {
        if(moveProgress >= 1.0f) {
            renderCell = Point2f((float)cell.second, (float)cell.first);
            return;
        }

        moveProgress += moveSpeed;
        if(moveProgress >= 1.0f) {
            moveProgress = 1.0f;
            cell = moveToCell;
        }

        float t = clampT(moveProgress, 0.0f, 1.0f);
        float rr = (float)moveFromCell.first  + (float)(moveToCell.first  - moveFromCell.first)  * t;
        float cc = (float)moveFromCell.second + (float)(moveToCell.second - moveFromCell.second) * t;
        renderCell = Point2f(cc, rr);
    }
};

// ============================================================
// AGENT LOGIC
// ============================================================
enum GoalType {
    GOAL_SLEEP,
    GOAL_BREAKFAST,
    GOAL_WORK,
    GOAL_LUNCH,
    GOAL_DINNER,
    GOAL_BATHROOM,
    GOAL_RELAX_HOME,
    GOAL_PARK,
    GOAL_SEA_BATH,
    GOAL_SCHOOL,
    GOAL_HIGHSCHOOL,
    GOAL_UNIVERSITY,
    GOAL_RETIRED_BEACH
};

enum DayRole {
    ROLE_CHILD_SCHOOL,
    ROLE_TEEN_HIGHSCHOOL,
    ROLE_YOUNG_UNIVERSITY,
    ROLE_ADULT_WORK,
    ROLE_RETIRED
};

struct Persona {
    int id = 0;
    int age = 0;

    SmoothMover mover;

    int homeDistrict = -1;

    IPoint bedroom;
    IPoint workplace;
    IPoint studyplace;

    vector<IPoint> localKitchens;
    vector<IPoint> localLivings;
    vector<IPoint> localWCs;
    vector<IPoint> localParks;
    vector<IPoint> localSea;

    vector<IPoint> path;
    IPoint target;

    GoalType currentGoal = GOAL_RELAX_HOME;
    string currentGoalLabel = "Casa";

    double hunger = 20.0;
    double bladder = 10.0;
    double energy = 80.0;
    double social = 60.0;

    int stayTimer = 0;
    int lastMealHour = -10;
    int lastBathroomHour = -10;

    double moveSmoothing = 0.22;
    int colorVariant = 0;
    bool maleStyle = true;

    float facingAngle = 0.0f;
    float desiredFacingAngle = 0.0f;

    bool returningFromBathroom = false;
    GoalType resumeGoalAfterBathroom = GOAL_RELAX_HOME;
    IPoint resumeTargetAfterBathroom = {0,0};

    IPoint cell() const { return mover.cell; }
    Point2f renderCell() const { return mover.renderCell; }

    void init(int pid, const IPoint &start) {
        id = pid;
        colorVariant = pid % 8;
        maleStyle = (pid % 2 == 0);
        moveSmoothing = 0.14 + frand01() * 0.08;
        mover.initAt(start, (float)moveSmoothing);
        target = start;
        facingAngle = (float)(frand01() * CV_PI * 2.0);
        desiredFacingAngle = facingAngle;
    }

    DayRole getRole() const {
        if(age >= 1 && age <= 12) return ROLE_CHILD_SCHOOL;
        if(age >= 13 && age <= 18) return ROLE_TEEN_HIGHSCHOOL;
        if(age >= 19 && age <= 22) return ROLE_YOUNG_UNIVERSITY;
        if(age >= 23 && age <= 65) return ROLE_ADULT_WORK;
        return ROLE_RETIRED;
    }

    void updateNeeds(long curTime) {
        int hour = getHour(curTime);

        hunger += 0.10;
        bladder += 0.11;

        if(hour >= 23 || hour < 7) energy += 0.26;
        else if(currentGoal == GOAL_WORK || currentGoal == GOAL_SCHOOL || currentGoal == GOAL_HIGHSCHOOL || currentGoal == GOAL_UNIVERSITY)
            energy -= 0.08;
        else if(currentGoal == GOAL_PARK || currentGoal == GOAL_SEA_BATH || currentGoal == GOAL_RETIRED_BEACH)
            energy -= 0.06;
        else
            energy -= 0.04;

        if(currentGoal == GOAL_RELAX_HOME || currentGoal == GOAL_PARK || currentGoal == GOAL_SEA_BATH || currentGoal == GOAL_RETIRED_BEACH)
            social += 0.03;
        else
            social -= 0.01;

        hunger = clampT(hunger, 0.0, 100.0);
        bladder = clampT(bladder, 0.0, 100.0);
        energy = clampT(energy, 0.0, 100.0);
        social = clampT(social, 0.0, 100.0);
    }

    GoalType chooseRoutineGoal(long curTime) {
        int hour = getHour(curTime);
        bool weekend = isWeekend(curTime);
        DayRole role = getRole();

        if(energy < 15.0 || hour >= 23 || hour < 7) return GOAL_SLEEP;

        if(hour >= 7 && hour < 9) {
            if(lastMealHour != hour && hunger > 18.0) return GOAL_BREAKFAST;
            return GOAL_RELAX_HOME;
        }

        if(!weekend && hour >= 9 && hour < 13) {
            switch(role) {
                case ROLE_CHILD_SCHOOL:     return GOAL_SCHOOL;
                case ROLE_TEEN_HIGHSCHOOL:  return GOAL_HIGHSCHOOL;
                case ROLE_YOUNG_UNIVERSITY: return GOAL_UNIVERSITY;
                case ROLE_ADULT_WORK:       return GOAL_WORK;
                case ROLE_RETIRED:          return GOAL_RETIRED_BEACH;
            }
        }

        if(hour >= 13 && hour < 15) {
            if(lastMealHour != hour) return GOAL_LUNCH;
            return GOAL_RELAX_HOME;
        }

        if(!weekend && hour >= 15 && hour < 19) {
            switch(role) {
                case ROLE_CHILD_SCHOOL:     return GOAL_SCHOOL;
                case ROLE_TEEN_HIGHSCHOOL:  return GOAL_HIGHSCHOOL;
                case ROLE_YOUNG_UNIVERSITY: return GOAL_UNIVERSITY;
                case ROLE_ADULT_WORK:       return GOAL_WORK;
                case ROLE_RETIRED:          return GOAL_RETIRED_BEACH;
            }
        }

        if(hour >= 20 && hour < 22) {
            if(lastMealHour != hour && hunger > 25.0) return GOAL_DINNER;
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

    GoalType chooseGoal(long curTime) {
        int hour = getHour(curTime);
        if(bladder > 90.0 && hour != lastBathroomHour) return GOAL_BATHROOM;
        return chooseRoutineGoal(curTime);
    }

    IPoint chooseGoalTarget(GoalType g) {
        switch(g) {
            case GOAL_SLEEP:         currentGoalLabel = "Dormir";      return bedroom;
            case GOAL_BREAKFAST:     currentGoalLabel = "Desayuno";    return nearestNearHome(bedroom, localKitchens);
            case GOAL_WORK:          currentGoalLabel = "Trabajo";     return workplace;
            case GOAL_SCHOOL:        currentGoalLabel = "School";      return studyplace;
            case GOAL_HIGHSCHOOL:    currentGoalLabel = "HighSchool";  return studyplace;
            case GOAL_UNIVERSITY:    currentGoalLabel = "Universidad"; return studyplace;
            case GOAL_RETIRED_BEACH: currentGoalLabel = "Playa";       return !localSea.empty() ? randomFromList(localSea) : bedroom;
            case GOAL_LUNCH:         currentGoalLabel = "Comida";      return nearestNearHome(bedroom, localKitchens);
            case GOAL_DINNER:        currentGoalLabel = "Cena";        return nearestNearHome(bedroom, localKitchens);
            case GOAL_BATHROOM:      currentGoalLabel = "WC";          return nearestBathroomFromCurrent(cell(), localWCs);
            case GOAL_SEA_BATH:      currentGoalLabel = "Bano mar";    return !localSea.empty() ? randomFromList(localSea) : bedroom;
            case GOAL_PARK:          currentGoalLabel = "Parque";      return !localParks.empty() ? randomFromList(localParks) : bedroom;
            case GOAL_RELAX_HOME:
            default:                 currentGoalLabel = "Salon";       return nearestNearHome(bedroom, localLivings);
        }
    }

    void rebuildPath(const vector<vector<Terrain>> &terrain) {
        path = getPath(terrain, cell(), target);
    }

    void interactAtDestination(long curTime) {
        int hour = getHour(curTime);

        switch(currentGoal) {
            case GOAL_SLEEP:
                energy += 1.5; hunger += 0.03; bladder += 0.03; stayTimer = 14 + irand(0, 12); break;
            case GOAL_BREAKFAST:
            case GOAL_LUNCH:
            case GOAL_DINNER:
                hunger -= 18.0; energy += 2.0; bladder += 1.0; lastMealHour = hour; stayTimer = 5 + irand(0, 4); break;
            case GOAL_WORK:
            case GOAL_SCHOOL:
            case GOAL_HIGHSCHOOL:
            case GOAL_UNIVERSITY:
                energy -= 0.8; hunger += 0.7; bladder += 0.5; stayTimer = 10 + irand(0, 10); break;
            case GOAL_BATHROOM:
                bladder -= 28.0; lastBathroomHour = hour; stayTimer = 2 + irand(0, 2); break;
            case GOAL_RETIRED_BEACH:
            case GOAL_SEA_BATH:
                social += 4.0; energy -= 0.8; hunger += 0.5; bladder += 0.4; stayTimer = 8 + irand(0, 10); break;
            case GOAL_PARK:
                social += 3.0; energy -= 0.6; stayTimer = 5 + irand(0, 6); break;
            case GOAL_RELAX_HOME:
            default:
                social += 1.2; energy += 0.7; stayTimer = 7 + irand(0, 8); break;
        }

        hunger = clampT(hunger, 0.0, 100.0);
        bladder = clampT(bladder, 0.0, 100.0);
        energy = clampT(energy, 0.0, 100.0);
        social = clampT(social, 0.0, 100.0);
    }

    void updateMovementOnly() {
        Point2f before = mover.renderCell;
        mover.update();
        Point2f after = mover.renderCell;

        Point2f delta = after - before;
        if(norm(delta) > 0.0005f) {
            desiredFacingAngle = atan2(delta.y, delta.x);
        }
        facingAngle = lerpAngle(facingAngle, desiredFacingAngle, 0.18f);
    }

    void stepDecision(const vector<vector<Terrain>> &terrain, long curTime) {
        updateNeeds(curTime);

        if(!mover.canStartNewStep()) return;
        if(stayTimer > 0) {
            stayTimer--;
            return;
        }

        if(currentGoal == GOAL_BATHROOM && path.empty()) {
            interactAtDestination(curTime);

            if(returningFromBathroom) {
                currentGoal = resumeGoalAfterBathroom;
                target = resumeTargetAfterBathroom;
                returningFromBathroom = false;
                rebuildPath(terrain);
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
            if(norm(delta) > 0.001f) {
                desiredFacingAngle = atan2(delta.y, delta.x);
            }

            mover.startMoveTo(next);
        } else {
            interactAtDestination(curTime);
        }
    }
};

// ============================================================
// PLAYER
// ============================================================
struct Player {
    SmoothMover mover;
    float facingAngle = 0.0f;
    float desiredFacingAngle = 0.0f;

    int colorVariant = 7;
    bool maleStyle = true;

    void init(const IPoint &start) {
        mover.initAt(start, 0.18f);
        facingAngle = 0.0f;
        desiredFacingAngle = 0.0f;
        colorVariant = 7;
        maleStyle = true;
    }

    IPoint cell() const { return mover.cell; }
    Point2f renderCell() const { return mover.renderCell; }

    void update() {
        Point2f before = mover.renderCell;
        mover.update();
        Point2f after = mover.renderCell;

        Point2f delta = after - before;
        if(norm(delta) > 0.0005f) {
            desiredFacingAngle = atan2(delta.y, delta.x);
        }
        facingAngle = lerpAngle(facingAngle, desiredFacingAngle, 0.22f);
    }

    bool tryMove(int dr, int dc, const vector<vector<Terrain>> &terrain) {
        if(!mover.canStartNewStep()) return false;

        IPoint cur = cell();
        IPoint nxt = {cur.first + dr, cur.second + dc};

        int H = (int)terrain.size();
        int W = (int)terrain[0].size();

        if(nxt.first < 0 || nxt.first >= H || nxt.second < 0 || nxt.second >= W) return false;
        if(!isWalkable(terrain[nxt.first][nxt.second])) return false;

        Point2f delta((float)dc, (float)dr);
        if(norm(delta) > 0.001f) {
            desiredFacingAngle = atan2(delta.y, delta.x);
        }

        mover.startMoveTo(nxt);
        return true;
    }
};

// ============================================================
// CAMERA
// ============================================================
struct Camera2D {
    float centerX = 0.0f;
    float centerY = 0.0f;
    float zoom = 1.0f;

    float targetCenterX = 0.0f;
    float targetCenterY = 0.0f;
    float targetZoom = 1.0f;

    float manualZoom = CAMERA_NPC_ZOOM_FACTOR;

    float orbitYaw = (float)(45.0 * CV_PI / 180.0);
    float targetOrbitYaw = (float)(45.0 * CV_PI / 180.0);

    float tilt = 1.0f;
    float targetTilt = 1.0f;

    int focusedNpc = 0;
    bool followPlayer = true;

    void changeZoom(float delta) {
        manualZoom = clampT(manualZoom + delta, CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX);
        targetZoom = manualZoom;
    }

    void changeOrbitDeg(float deltaDeg) {
        targetOrbitYaw = normalizeAngle(targetOrbitYaw + (float)(deltaDeg * CV_PI / 180.0));
    }

    void changeTilt(float delta) {
        targetTilt = clampT(targetTilt + delta, CAMERA_TILT_MIN, CAMERA_TILT_MAX);
    }

    void nextNpcFocus(const vector<Persona> &agents) {
        if(agents.empty()) return;
        focusedNpc = (focusedNpc + 1) % (int)agents.size();
        followPlayer = false;
    }

    void updateFollowPoint(float rx, float ry) {
        targetCenterX = rx;
        targetCenterY = ry;
        targetZoom = manualZoom;

        centerX += (targetCenterX - centerX) * CAMERA_CENTER_SMOOTH;
        centerY += (targetCenterY - centerY) * CAMERA_CENTER_SMOOTH;
        zoom    += (targetZoom    - zoom)    * CAMERA_ZOOM_SMOOTH;

        orbitYaw = lerpAngle(orbitYaw, targetOrbitYaw, CAMERA_ANGLE_SMOOTH);
        tilt += (targetTilt - tilt) * CAMERA_ANGLE_SMOOTH;
    }

    void updatePlayerOrCinematic(const Player &player,
                                 const vector<Persona> &agents,
                                 long long nowRealMs,
                                 long long lastPlayerInputMs)
    {
        bool playerRecentlyMoved = (nowRealMs - lastPlayerInputMs) < PLAYER_IDLE_FALLBACK_MS;

        if(playerRecentlyMoved || agents.empty()) {
            followPlayer = true;
            updateFollowPoint(player.renderCell().x, player.renderCell().y);
            return;
        }

        followPlayer = false;
        const Persona &a = agents[focusedNpc];
        updateFollowPoint(a.renderCell().x, a.renderCell().y);
    }
};

// ============================================================
// PROJECTION / DEPTH
// ============================================================
float depthAt(float row, float col, const Camera2D &cam) {
    float x = col;
    float y = row;
    return x * sin(cam.orbitYaw) + y * cos(cam.orbitYaw);
}

Point2f projectWorldPoint(float row, float col, float zOffset, const Camera2D &cam) {
    float x = col;
    float y = row;

    float c = cos(cam.orbitYaw);
    float s = sin(cam.orbitYaw);

    float right = x * c - y * s;
    float depth = x * s + y * c;

    const float diag = 1.41421356237f;
    float sx = right * (TILE_W * 0.5f * diag);
    float sy = depth * (TILE_H * 0.5f * diag) * cam.tilt
             - zOffset * (0.68f + 0.32f * cam.tilt);

    return Point2f(sx, sy);
}

Point cameraProject(const Point2f &world, const Camera2D &cam, const Point2f &camWorldCenter) {
    float x = (world.x - camWorldCenter.x) * cam.zoom + SCREEN_W * 0.5f;
    float y = (world.y - camWorldCenter.y) * cam.zoom + SCREEN_H * 0.5f;
    return Point((int)round(x), (int)round(y));
}

Point2f tileWorldCenter(float row, float col, const vector<vector<float>> &zmap, const Camera2D &cam) {
    float z = sampleZBilinear(zmap, row, col);
    return projectWorldPoint(row, col, z, cam);
}

Scalar shadeColor(const Scalar &c, double f) {
    return Scalar(
        saturate_cast<uchar>(c[0] * f),
        saturate_cast<uchar>(c[1] * f),
        saturate_cast<uchar>(c[2] * f)
    );
}

Scalar terrainTopColor(Terrain t) {
    switch(t) {
        case ASPHALT:     return Scalar(90, 80, 105);
        case WALKABLE:    return Scalar(135, 135, 135);
        case WALL:        return Scalar(235, 235, 235);
        case PARK:        return Scalar(40, 155, 55);
        case WORK:        return Scalar(130, 130, 130);
        case WC:          return Scalar(60, 60, 220);
        case LIVING:      return Scalar(50, 220, 220);
        case BEDROOM:     return Scalar(220, 80, 80);
        case KITCHEN:     return Scalar(80, 220, 220);
        case SEAWATER:    return Scalar(194, 72, 76);
        case SCHOOL:      return Scalar(50, 132, 213);
        case HIGHSCHOOL:  return Scalar(48, 95, 142);
        case UNIVERSITY:  return Scalar(59, 85, 112);
        default:          return Scalar(20, 20, 20);
    }
}

void drawIsoPrism(Mat &img,
                  float row, float col,
                  int heightPx,
                  int heightOffsetPx,
                  const Scalar &topC,
                  const Camera2D &cam,
                  const Point2f &camCenter,
                  const vector<vector<float>> &zmap)
{
    float z = sampleZBilinear(zmap, row, col);
    Point2f c = projectWorldPoint(row, col, z, cam);

    float cYaw = cos(cam.orbitYaw);
    float sYaw = sin(cam.orbitYaw);

    Point2f rowVec = projectWorldPoint(row + 1.0f, col, z, cam) - c;
    Point2f colVec = projectWorldPoint(row, col + 1.0f, z, cam) - c;

    Point2f topP    = c - (rowVec + colVec) * 0.5f;
    Point2f rightP  = c + (colVec - rowVec) * 0.5f;
    Point2f bottomP = c + (rowVec + colVec) * 0.5f;
    Point2f leftP   = c + (rowVec - colVec) * 0.5f;

    int h = (int)round(heightPx * cam.zoom * (0.68f + 0.32f * cam.tilt));
    int yOffset = (int)round(heightOffsetPx * cam.zoom * (0.68f + 0.32f * cam.tilt));

    Point top    = cameraProject(topP, cam, camCenter);
    Point right  = cameraProject(rightP, cam, camCenter);
    Point bottom = cameraProject(bottomP, cam, camCenter);
    Point left   = cameraProject(leftP, cam, camCenter);

    // elevar cara superior
    top.y    -= yOffset;
    right.y  -= yOffset;
    bottom.y -= yOffset;
    left.y   -= yOffset;

    // extrusion hacia abajo
    Point rightB  = Point(right.x,  right.y + h);
    Point bottomB = Point(bottom.x, bottom.y + h);
    Point leftB   = Point(left.x,   left.y + h);

    float lightX = cYaw;
    float lightY = sYaw;

    Point2f leftNormal(-1.0f,  1.0f);
    Point2f rightNormal( 1.0f,  1.0f);

    float nl = leftNormal.x * lightX + leftNormal.y * lightY;
    float nr = rightNormal.x * lightX + rightNormal.y * lightY;

    double lf = 0.62 + 0.18 * nl;
    double rf = 0.58 + 0.18 * nr;

    lf = clampT(lf, 0.45, 0.92);
    rf = clampT(rf, 0.42, 0.88);

    Scalar leftC  = shadeColor(topC, lf);
    Scalar rightC = shadeColor(topC, rf);
    Scalar outline(20,20,20);

    Point leftFace[4]  = { left, bottom, bottomB, leftB };
    Point rightFace[4] = { right, bottom, bottomB, rightB };
    Point topFace[4]   = { top, right, bottom, left };

    fillConvexPoly(img, leftFace, 4, leftC, LINE_AA);
    fillConvexPoly(img, rightFace, 4, rightC, LINE_AA);
    fillConvexPoly(img, topFace, 4, topC, LINE_AA);

    polylines(img, vector<Point>{top, right, bottom, left}, true, outline, 1, LINE_AA);
    line(img, right, rightB, outline, 1, LINE_AA);
    line(img, bottom, bottomB, outline, 1, LINE_AA);
    line(img, left, leftB, outline, 1, LINE_AA);
}

void drawFloorTile(Mat &img,
                   int row, int col,
                   Terrain t,
                   const Camera2D &cam,
                   const Point2f &camCenter,
                   const vector<vector<float>> &zmap)
{
    Scalar topC = terrainTopColor(t);
    drawIsoPrism(img, (float)row, (float)col, FLOOR_THICKNESS, 0, topC, cam, camCenter, zmap);

    Point base = cameraProject(tileWorldCenter((float)row, (float)col, zmap, cam), cam, camCenter);

    if(t == WC) {
        putText(img, "WC", Point(base.x - 8, base.y + 4), FONT_HERSHEY_PLAIN,
                max(0.7, cam.zoom * 0.65), Scalar(255,255,255), 1, LINE_AA);
    } else if(t == BEDROOM) {
        int s = max(2, (int)round(3 * cam.zoom));
        rectangle(img, Rect(base.x - s, base.y - s/2, s*2, s), Scalar(255,255,255), FILLED, LINE_AA);
    } else if(t == KITCHEN) {
        int r = max(2, (int)round(2 * cam.zoom));
        circle(img, base, r, Scalar(255,255,255), FILLED, LINE_AA);
    } else if(t == WORK) {
        int s = max(2, (int)round(2 * cam.zoom));
        rectangle(img, Rect(base.x - s, base.y - s, s*2+1, s*2+1), Scalar(230,230,230), FILLED, LINE_AA);
    } else if(t == LIVING) {
        ellipse(img, base, Size(max(2,(int)round(3*cam.zoom)), max(1,(int)round(2*cam.zoom))), 0, 0, 360,
                Scalar(255,255,255), 1, LINE_AA);
    } else if(t == PARK) {
        circle(img, Point(base.x, base.y - max(3, (int)round(8 * cam.zoom))),
               max(2, (int)round(4 * cam.zoom)), Scalar(40,180,60), FILLED, LINE_AA);
        line(img, Point(base.x, base.y - max(2, (int)round(2 * cam.zoom))),
             Point(base.x, base.y - max(5, (int)round(7 * cam.zoom))),
             Scalar(40,80,120), 2, LINE_AA);
    } else if(t == SEAWATER) {
        int rr = max(3, (int)round(3.5 * cam.zoom));
        ellipse(img, Point(base.x, base.y + max(0, (int)round(1 * cam.zoom))),
                Size(rr + 2, max(2, rr / 2)), 0, 0, 360, Scalar(255, 210, 160), 1, LINE_AA);
        line(img, Point(base.x - rr, base.y), Point(base.x + rr, base.y),
             Scalar(255, 230, 190), 1, LINE_AA);
    } else if(t == SCHOOL) {
        putText(img, "SCH", Point(base.x - 10, base.y + 4), FONT_HERSHEY_PLAIN,
                max(0.7, cam.zoom * 0.55), Scalar(255,255,255), 1, LINE_AA);
    } else if(t == HIGHSCHOOL) {
        putText(img, "HS", Point(base.x - 8, base.y + 4), FONT_HERSHEY_PLAIN,
                max(0.7, cam.zoom * 0.60), Scalar(255,255,255), 1, LINE_AA);
    } else if(t == UNIVERSITY) {
        putText(img, "UNI", Point(base.x - 9, base.y + 4), FONT_HERSHEY_PLAIN,
                max(0.7, cam.zoom * 0.55), Scalar(255,255,255), 1, LINE_AA);
    }
}

void drawWallTile(Mat &img,
                  int row, int col,
                  const Camera2D &cam,
                  const Point2f &camCenter,
                  const vector<vector<float>> &zmap)
{
    Scalar wallTop = terrainTopColor(WALL);
    drawIsoPrism(img, (float)row, (float)col, WALL_HEIGHT, WALL_HEIGHT_OFFSET, wallTop, cam, camCenter, zmap);
}

Point toScreenLocal(const Point &origin, const Point2f &local, float scale = 1.0f) {
    return Point((int)round(origin.x + local.x * scale),
                 (int)round(origin.y + local.y * scale));
}

void drawFilledPolygon(Mat &img, const vector<Point> &pts, const Scalar &color) {
    if(pts.size() >= 3) fillConvexPoly(img, pts, color, LINE_AA);
}

void drawHumanDirectionalAt(Mat &img,
                            const Point2f &renderCell,
                            float facingAngle,
                            int colorVariant,
                            bool maleStyle,
                            const Camera2D &cam,
                            const Point2f &camCenter,
                            const vector<vector<float>> &zmap)
{
    Point2f world = tileWorldCenter(renderCell.y, renderCell.x, zmap, cam);
    Point base = cameraProject(world, cam, camCenter);

    float z = cam.zoom;
    float scale = max(0.8f, z);

    Point shadowCenter(base.x, base.y + (int)round(2 * scale));
    ellipse(img, shadowCenter,
            Size(max(3, (int)round(5 * scale)), max(2, (int)round(2.5f * scale))),
            0, 0, 360, Scalar(0, 0, 0), FILLED, LINE_AA);

    Scalar skin(190, 210, 235);
    Scalar outfit[8] = {
        Scalar(30, 70, 220),
        Scalar(40, 180, 220),
        Scalar(220, 160, 40),
        Scalar(180, 60, 180),
        Scalar(50, 170, 80),
        Scalar(90, 90, 90),
        Scalar(70, 120, 240),
        Scalar(40, 120, 200)
    };
    Scalar body = outfit[colorVariant % 8];
    Scalar bodyDark = shadeColor(body, 0.72);
    Scalar bodyLight = shadeColor(body, 1.18);
    Scalar legs(35, 35, 35);
    Scalar shoe(20, 20, 20);
    Scalar hair = maleStyle ? Scalar(30, 50, 70) : Scalar(20, 30, 40);

    float viewAdjustedFacing = facingAngle - cam.orbitYaw + (float)(CV_PI * 0.25);
    Point2f forward(cos(viewAdjustedFacing), sin(viewAdjustedFacing));
    Point2f right(forward.y, -forward.x);

    float headR = 3.8f * scale;
    float torsoH = 10.0f * scale;
    float torsoW = 5.5f * scale;
    float hipW = 4.2f * scale;
    float legH = 8.0f * scale;
    float shoulderW = 6.0f * scale;
    float armLen = 6.0f * scale;

    Point torsoBase(base.x, base.y - (int)round(6 * scale));

    float frontness = forward.y;
    float sideness = forward.x;
    float widthFactor = 0.70f + 0.35f * fabs(sideness);
    float depthFactor = 0.65f + 0.35f * fabs(frontness);

    Point2f bodyUp(0.0f, -torsoH);
    Point2f bodyRight = right * (torsoW * widthFactor);
    Point2f bodyFront = forward * (1.6f * depthFactor);

    vector<Point2f> torsoLocalF = {
        -bodyRight * 0.55f + bodyFront * 0.35f,
         bodyRight * 0.55f + bodyFront * 0.35f,
         bodyRight * 0.40f - bodyUp - bodyFront * 0.25f,
        -bodyRight * 0.40f - bodyUp - bodyFront * 0.25f
    };

    vector<Point> torsoPoly;
    for(const auto &p : torsoLocalF) torsoPoly.push_back(toScreenLocal(torsoBase, p));

    Scalar torsoColor = (frontness >= 0.0f) ? bodyLight : bodyDark;
    drawFilledPolygon(img, torsoPoly, torsoColor);

    Point2f neckLocal = -bodyUp + bodyFront * 0.05f;
    Point headCenter = toScreenLocal(torsoBase, neckLocal + Point2f(0.0f, -headR * 1.1f));

    ellipse(img, headCenter,
            Size(max(2, (int)round(headR * (0.95f + 0.20f * fabs(sideness)))),
                 max(2, (int)round(headR * 1.10f))),
            0, 0, 360, skin, FILLED, LINE_AA);

    Point hairCenter = Point(headCenter.x, headCenter.y - (int)round(headR * 0.45f));
    ellipse(img, hairCenter,
            Size(max(2, (int)round(headR * (0.95f + 0.20f * fabs(sideness)))),
                 max(2, (int)round(headR * 0.70f))),
            0, 0, 360, hair, FILLED, LINE_AA);

    if(fabs(sideness) > 0.20f) {
        int eyeOffsetX = (int)round((sideness > 0 ? 1.2f : -1.2f) * scale);
        circle(img, Point(headCenter.x + eyeOffsetX, headCenter.y - (int)round(0.5f * scale)),
               max(1, (int)round(0.6f * scale)), Scalar(40,40,40), FILLED, LINE_AA);
    } else if(frontness > 0.10f) {
        circle(img, Point(headCenter.x - (int)round(1.2f * scale), headCenter.y - (int)round(0.4f * scale)),
               max(1, (int)round(0.55f * scale)), Scalar(40,40,40), FILLED, LINE_AA);
        circle(img, Point(headCenter.x + (int)round(1.2f * scale), headCenter.y - (int)round(0.4f * scale)),
               max(1, (int)round(0.55f * scale)), Scalar(40,40,40), FILLED, LINE_AA);
    }

    Point2f hipCenter(0.0f, 0.0f);
    Point2f leftHip  = -right * (hipW * 0.45f) + forward * 0.3f * scale;
    Point2f rightHip =  right * (hipW * 0.45f) - forward * 0.3f * scale;

    Point2f leftFoot  = leftHip + Point2f(0.0f, legH);
    Point2f rightFoot = rightHip + Point2f(0.0f, legH);

    line(img, toScreenLocal(torsoBase, hipCenter + leftHip),
              toScreenLocal(torsoBase, hipCenter + leftFoot), legs, max(1, (int)round(1.6f * scale)), LINE_AA);
    line(img, toScreenLocal(torsoBase, hipCenter + rightHip),
              toScreenLocal(torsoBase, hipCenter + rightFoot), legs, max(1, (int)round(1.6f * scale)), LINE_AA);

    ellipse(img, toScreenLocal(torsoBase, hipCenter + leftFoot),
            Size(max(1, (int)round(1.5f * scale)), max(1, (int)round(0.9f * scale))),
            0, 0, 360, shoe, FILLED, LINE_AA);
    ellipse(img, toScreenLocal(torsoBase, hipCenter + rightFoot),
            Size(max(1, (int)round(1.5f * scale)), max(1, (int)round(0.9f * scale))),
            0, 0, 360, shoe, FILLED, LINE_AA);

    Point2f shoulderCenter = -bodyUp * 0.63f;
    Point2f leftShoulder  = shoulderCenter - right * (shoulderW * 0.50f);
    Point2f rightShoulder = shoulderCenter + right * (shoulderW * 0.50f);

    Point2f leftHand  = leftShoulder  + Point2f(0.0f, armLen * 0.72f);
    Point2f rightHand = rightShoulder + Point2f(0.0f, armLen * 0.72f);

    line(img, toScreenLocal(torsoBase, leftShoulder),
              toScreenLocal(torsoBase, leftHand), bodyDark, max(1, (int)round(1.4f * scale)), LINE_AA);
    line(img, toScreenLocal(torsoBase, rightShoulder),
              toScreenLocal(torsoBase, rightHand), bodyDark, max(1, (int)round(1.4f * scale)), LINE_AA);

    circle(img, toScreenLocal(torsoBase, leftHand),  max(1, (int)round(0.9f * scale)), skin, FILLED, LINE_AA);
    circle(img, toScreenLocal(torsoBase, rightHand), max(1, (int)round(0.9f * scale)), skin, FILLED, LINE_AA);
}

void drawCrosshairCorners(Mat &img, Point center, int size, int seg, const Scalar &color, int thickness) {
    int x = center.x;
    int y = center.y;
    line(img, Point(x - size, y - size), Point(x - size + seg, y - size), color, thickness, LINE_AA);
    line(img, Point(x - size, y - size), Point(x - size, y - size + seg), color, thickness, LINE_AA);

    line(img, Point(x + size, y - size), Point(x + size - seg, y - size), color, thickness, LINE_AA);
    line(img, Point(x + size, y - size), Point(x + size, y - size + seg), color, thickness, LINE_AA);

    line(img, Point(x - size, y + size), Point(x - size + seg, y + size), color, thickness, LINE_AA);
    line(img, Point(x - size, y + size), Point(x - size, y + size - seg), color, thickness, LINE_AA);

    line(img, Point(x + size, y + size), Point(x + size - seg, y + size), color, thickness, LINE_AA);
    line(img, Point(x + size, y + size), Point(x + size, y + size - seg), color, thickness, LINE_AA);
}

void drawPlayer(Mat &img,
                const Player &p,
                const Camera2D &cam,
                const Point2f &camCenter,
                const vector<vector<float>> &zmap)
{
    drawHumanDirectionalAt(img, p.renderCell(), p.facingAngle, p.colorVariant, p.maleStyle, cam, camCenter, zmap);

    Point2f world = tileWorldCenter(p.renderCell().y, p.renderCell().x, zmap, cam);
    Point base = cameraProject(world, cam, camCenter);

    int sz = max(8, (int)round(13 * cam.zoom));
    int seg = max(3, (int)round(5 * cam.zoom));
    int th = max(1, (int)round(2 * cam.zoom));

    drawCrosshairCorners(
        img,
        Point(base.x, base.y - (int)round(10 * cam.zoom)),
        sz, seg,
        Scalar(30, 30, 255),
        th
    );
}

// ============================================================
// RENDER
// ============================================================
struct DrawItem {
    int kind; // 0=floor/wall, 1=agent, 2=player
    int row = 0;
    int col = 0;
    int agentIndex = -1;
    float depth = 0.0f;
};

void drawScene(Mat &frame,
               const vector<vector<Terrain>> &terrain,
               const vector<vector<float>> &zmap,
               const vector<Persona> &agents,
               const Player &player,
               const Camera2D &cam)
{
    frame = Mat(SCREEN_H, SCREEN_W, CV_8UC3, Scalar(18, 18, 24));

    Point2f camCenterWorld = tileWorldCenter(cam.centerY, cam.centerX, zmap, cam);

    int H = (int)terrain.size();
    int W = (int)terrain[0].size();

    vector<DrawItem> items;
    items.reserve(H * W + (int)agents.size() + 1);

    for(int row = 0; row < H; ++row) {
        for(int col = 0; col < W; ++col) {
            Terrain t = terrain[row][col];
            if(t == VOID_TERRAIN) continue;

            DrawItem di;
            di.kind = 0;
            di.row = row;
            di.col = col;
            di.depth = depthAt((float)row, (float)col, cam);
            items.push_back(di);
        }
    }

    for(int i = 0; i < (int)agents.size(); ++i) {
        DrawItem di;
        di.kind = 1;
        di.agentIndex = i;
        di.depth = depthAt(agents[i].renderCell().y, agents[i].renderCell().x, cam) + 0.15f;
        items.push_back(di);
    }

    {
        DrawItem di;
        di.kind = 2;
        di.depth = depthAt(player.renderCell().y, player.renderCell().x, cam) + 0.16f;
        items.push_back(di);
    }

    stable_sort(items.begin(), items.end(), [](const DrawItem &a, const DrawItem &b) {
        return a.depth < b.depth;
    });

    for(const auto &it : items) {
        if(it.kind == 0) {
            Terrain t = terrain[it.row][it.col];
            if(t == WALL) drawWallTile(frame, it.row, it.col, cam, camCenterWorld, zmap);
            else drawFloorTile(frame, it.row, it.col, t, cam, camCenterWorld, zmap);
        } else if(it.kind == 1) {
            const auto &a = agents[it.agentIndex];
            drawHumanDirectionalAt(frame, a.renderCell(), a.facingAngle, a.colorVariant, a.maleStyle, cam, camCenterWorld, zmap);
        } else {
            drawPlayer(frame, player, cam, camCenterWorld, zmap);
        }
    }
}

bool isArrowLeft(int key)  { return key == 81 || key == 2424832 || key == 65361; }
bool isArrowUp(int key)    { return key == 82 || key == 2490368 || key == 65362; }
bool isArrowRight(int key) { return key == 83 || key == 2555904 || key == 65363; }
bool isArrowDown(int key)  { return key == 84 || key == 2621440 || key == 65364; }

// ============================================================
// MAIN
// ============================================================
int main() {
    Mat img = imread("casas.png");
    if(img.empty()) {
        printf("Failed to load casas.png\n");
        return -1;
    }

    Mat zimg = imread("z.png");
    if(zimg.empty()) {
        printf("Failed to load z.png\n");
        return -1;
    }

    if(img.size() != zimg.size()) {
        printf("casas.png and z.png must have exactly the same resolution\n");
        return -1;
    }

    if(img.cols != MAP_W_EXPECTED || img.rows != MAP_H_EXPECTED) {
        printf("Warning: casas.png is %dx%d, expected around %dx%d\n",
               img.cols, img.rows, MAP_W_EXPECTED, MAP_H_EXPECTED);
    }

    auto terrain = buildTerrain(img);
    auto zmap = buildZOffsets(zimg, Z_HEIGHT_MULTIPLIER);

    int H = (int)terrain.size();
    int W = (int)terrain[0].size();

    vector<IPoint> walkables, bedrooms, works, kitchens, wcs, livings, parks, seawaters, schools, highschools, universities;
    for(int y = 0; y < H; ++y) {
        for(int x = 0; x < W; ++x) {
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
            if(!availableWorks.empty()) {
                a.workplace = availableWorks.back();
                availableWorks.pop_back();
            } else if(!works.empty()) {
                a.workplace = randomFromList(works);
            } else {
                a.workplace = bedroom;
            }
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
    cam.orbitYaw = (float)(45.0 * CV_PI / 180.0);
    cam.targetOrbitYaw = cam.orbitYaw;
    cam.tilt = 1.0f;
    cam.targetTilt = 1.0f;

    namedWindow("Sim", WINDOW_NORMAL);
    resizeWindow("Sim", SCREEN_W, SCREEN_H);

    long curTime = 7 * 3600;
    Mat frame;

    long long lastPlayerInputMs = nowMs();

    while(true) {
        long long currentRealMs = nowMs();

        int key = waitKeyEx(16);
        if(key == 27) break;

        if(key == '+' || key == '=') cam.changeZoom(CAMERA_ZOOM_STEP);
        if(key == '-' || key == '_') cam.changeZoom(-CAMERA_ZOOM_STEP);
        if(key == 171 || key == 107) cam.changeZoom(CAMERA_ZOOM_STEP);
        if(key == 173 || key == 109) cam.changeZoom(-CAMERA_ZOOM_STEP);

        // izquierda/derecha = orbita horizontal
        if(isArrowLeft(key))  cam.changeOrbitDeg(-CAMERA_ORBIT_STEP_DEG);
        if(isArrowRight(key)) cam.changeOrbitDeg( CAMERA_ORBIT_STEP_DEG);

        // arriba/abajo = orbita vertical
        if(isArrowUp(key))    cam.changeTilt( CAMERA_TILT_STEP);
        if(isArrowDown(key))  cam.changeTilt(-CAMERA_TILT_STEP);

        // espacio = cambia foco npc
        if(key == 32) {
            cam.nextNpcFocus(agents);
        }

        bool movedPlayer = false;

        if(key == 'w' || key == 'W') movedPlayer = player.tryMove(-1, 0, terrain);
        if(key == 's' || key == 'S') movedPlayer = player.tryMove( 1, 0, terrain);
        if(key == 'a' || key == 'A') movedPlayer = player.tryMove( 0,-1, terrain);
        if(key == 'd' || key == 'D') movedPlayer = player.tryMove( 0, 1, terrain);

        if(movedPlayer) {
            lastPlayerInputMs = currentRealMs;
            cam.followPlayer = true;
        }

        for(auto &a : agents) {
            a.stepDecision(terrain, curTime);
        }

        for(auto &a : agents) {
            a.updateMovementOnly();
        }

        player.update();

        cam.updatePlayerOrCinematic(player, agents, currentRealMs, lastPlayerInputMs);

        drawScene(frame, terrain, zmap, agents, player, cam);

        rectangle(frame, Rect(20, 20, 1560, 190), Scalar(0,0,0), FILLED);
        rectangle(frame, Rect(20, 20, 1560, 190), Scalar(95,95,95), 1);

        putText(frame, secondsToDateTimeStr(curTime), Point(35, 55),
                FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255,255,255), 2, LINE_AA);

        ostringstream oss;
        oss << "Agentes: " << agents.size()
            << "  Camara: " << (cam.followPlayer ? "PLAYER" : "NPC " + to_string(cam.focusedNpc))
            << "  Zoom: " << fixed << setprecision(2) << cam.zoom
            << "  Tilt: " << fixed << setprecision(2) << cam.tilt
            << "  OrbitDeg: " << fixed << setprecision(1) << (cam.orbitYaw * 180.0 / CV_PI)
            << "  WallHeight: " << WALL_HEIGHT
            << "  WallOffset: " << WALL_HEIGHT_OFFSET
            << "  Zmult: " << fixed << setprecision(2) << Z_HEIGHT_MULTIPLIER;
        putText(frame, oss.str(), Point(35, 85),
                FONT_HERSHEY_SIMPLEX, 0.60, Scalar(220,220,220), 1, LINE_AA);

        long long idleLeftMs = max(0LL, PLAYER_IDLE_FALLBACK_MS - (currentRealMs - lastPlayerInputMs));
        ostringstream pss;
        pss << "WASD mueve | Flechas izq/der orbitan horizontal | Flechas arriba/abajo orbitan vertical | Espacio cambia NPC | Numpad +/- zoom | fallback camara en "
            << fixed << setprecision(1) << (idleLeftMs / 1000.0) << " s";
        putText(frame, pss.str(), Point(35, 110),
                FONT_HERSHEY_SIMPLEX, 0.50, Scalar(180,220,255), 1, LINE_AA);

        float playerZ = sampleZBilinear(zmap, player.renderCell().y, player.renderCell().x);
        ostringstream ppos;
        ppos << "Player cell(" << player.cell().first << "," << player.cell().second << ")"
             << " z=" << fixed << setprecision(2) << playerZ;
        putText(frame, ppos.str(), Point(35, 135),
                FONT_HERSHEY_SIMPLEX, 0.50, Scalar(255,210,210), 1, LINE_AA);

        putText(frame, "z.png: canal R | 127 neutro | oscuro baja | claro sube",
                Point(35, 160),
                FONT_HERSHEY_SIMPLEX, 0.50, Scalar(220,255,220), 1, LINE_AA);

        int infoY = 230;
        for(int i = 0; i < min(6, (int)agents.size()); ++i) {
            const auto &a = agents[i];
            float az = sampleZBilinear(zmap, a.renderCell().y, a.renderCell().x);

            ostringstream line;
            line << "P" << a.id
                 << " edad:" << a.age
                 << " objetivo:" << a.currentGoalLabel
                 << " cell(" << a.cell().first << "," << a.cell().second << ")"
                 << " z:" << fixed << setprecision(1) << az;

            putText(frame, line.str(), Point(20, infoY),
                    FONT_HERSHEY_SIMPLEX, 0.50, Scalar(255,255,255), 1, LINE_AA);
            infoY += 22;
        }

        imshow("Sim", frame);

        curTime += timeStepSeconds;
    }

    return 0;
}
