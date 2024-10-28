
#ifndef TECO_PASSBUNDLES_H
#define TECO_PASSBUNDLES_H

#include "TeCo/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace teco {
#define GEN_PASS_DECL
#include "TeCo/PassBundles.h.inc"

#define GEN_PASS_REGISTRATION
#include "TeCo/PassBundles.h.inc"
} // namespace teco
} // namespace mlir

#endif // TECO_PASSBUNDLES_H
