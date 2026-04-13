#pragma once
#include "config.hpp"
#include "types.hpp"

__device__ __forceinline__ float3 mul(const Mat3& R, float3 v){
    return make_float3(
        R.m[0]*v.x + R.m[1]*v.y + R.m[2]*v.z,
        R.m[3]*v.x + R.m[4]*v.y + R.m[5]*v.z,
        R.m[6]*v.x + R.m[7]*v.y + R.m[8]*v.z
    );
}

__device__ bool project_point_gpu(float3 point3D, int& u, int& v, float& depth_out, float3 camPos, Mat3 R){
    float3 rel = make_float3(point3D.x - camPos.x, point3D.y - camPos.y, point3D.z - camPos.z);
    float3 rot = mul(R, rel);

    float z = rot.z;
    if (z <= 0.01f) return false;

    u = (int)(cfg::width  * 0.5f + cfg::focal_length * rot.x / z);
    v = (int)(cfg::height * 0.5f - cfg::focal_length * rot.y / z);

    if ((unsigned)u >= (unsigned)cfg::width || (unsigned)v >= (unsigned)cfg::height) return false;

    depth_out = z;
    return true;
}
