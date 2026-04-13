#include "biome.hpp"
#include "host_utils.hpp"

BiomeStyle biome_style_desert(){   return {0.95f, 0.10f, 0.30f, 0.95f, 0.20f}; }
BiomeStyle biome_style_valley(){   return {0.55f, 0.85f, 0.20f, 0.75f, 0.75f}; }
BiomeStyle biome_style_mountain(){ return {0.35f, 0.35f, 0.95f, 0.45f, 0.45f}; }
BiomeStyle biome_style_snow(){     return {0.10f, 0.45f, 0.80f, 0.10f, 0.65f}; }

BiomeStyle biome_lerp(const BiomeStyle& a, const BiomeStyle& b, float t){
    BiomeStyle r{};
    r.global_temperature = lerp_host(a.global_temperature, b.global_temperature, t);
    r.global_humidity    = lerp_host(a.global_humidity,    b.global_humidity,    t);
    r.global_ruggedness  = lerp_host(a.global_ruggedness,  b.global_ruggedness,  t);
    r.global_snowline    = lerp_host(a.global_snowline,    b.global_snowline,    t);
    r.cloudiness         = lerp_host(a.cloudiness,         b.cloudiness,         t);
    return r;
}

BiomeStyle random_biome_style(std::mt19937& rng){
    std::uniform_int_distribution<int> dist(0, 3);
    switch (dist(rng)) {
        case 0: return biome_style_desert();
        case 1: return biome_style_valley();
        case 2: return biome_style_mountain();
        default:return biome_style_snow();
    }
}
