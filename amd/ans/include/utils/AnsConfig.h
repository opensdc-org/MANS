#pragma once

namespace pans_hip {

// Logical ANS warp width. This value also determines the serialized warp-state
// layout, so all host and device code must use the same setting.
constexpr int kWarpSize = 64;

} // namespace pans_hip
