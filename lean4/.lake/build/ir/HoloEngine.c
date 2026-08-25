// Lean compiler output
// Module: HoloEngine
// Imports: public import Init public meta import Init public import HoloEngine.DualScale public import HoloEngine.TopoStability public import HoloEngine.FourierStateZ3 public import HoloEngine.PenroseFormalism
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_HoloEngine_HoloEngine_DualScale(uint8_t builtin);
lean_object* initialize_HoloEngine_HoloEngine_TopoStability(uint8_t builtin);
lean_object* initialize_HoloEngine_HoloEngine_FourierStateZ3(uint8_t builtin);
lean_object* initialize_HoloEngine_HoloEngine_PenroseFormalism(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_HoloEngine_HoloEngine(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_HoloEngine_HoloEngine_DualScale(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_HoloEngine_HoloEngine_TopoStability(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_HoloEngine_HoloEngine_FourierStateZ3(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_HoloEngine_HoloEngine_PenroseFormalism(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
