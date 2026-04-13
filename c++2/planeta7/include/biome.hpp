#pragma once
#include <random>
#include "types.hpp"

BiomeStyle biome_style_desert();
BiomeStyle biome_style_valley();
BiomeStyle biome_style_mountain();
BiomeStyle biome_style_snow();

BiomeStyle biome_lerp(const BiomeStyle& a, const BiomeStyle& b, float t);
BiomeStyle random_biome_style(std::mt19937& rng);
