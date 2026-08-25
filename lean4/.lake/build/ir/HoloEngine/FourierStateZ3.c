// Lean compiler output
// Module: HoloEngine.FourierStateZ3
// Imports: public import Init public meta import Init public import Mathlib.Data.Complex.Basic public import Mathlib.Data.Real.Basic public import Mathlib.Algebra.BigOperators.Group.Finset.Basic public import Mathlib.Algebra.BigOperators.Ring.Finset
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
lean_object* l_Int_pow(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* l_Int_add___boxed(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lean_int_neg(lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__0 = (const lean_object*)&lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__0_value;
static lean_once_cell_t lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0___redArg(lean_object*, lean_object*);
static lean_once_cell_t lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___closed__0;
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq(lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_zero__mode(lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_zero__mode___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_negWavevector(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___lam__0(lean_object* v_k_1_, lean_object* v_i_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = lean_apply_1(v_k_1_, v_i_2_);
v___x_4_ = lean_unsigned_to_nat(2u);
v___x_5_ = l_Int_pow(v___x_3_, v___x_4_);
lean_dec(v___x_3_);
return v___x_5_;
}
}
static lean_object* _init_lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_unsigned_to_nat(0u);
v___x_8_ = lean_nat_to_int(v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0(lean_object* v_s_9_){
_start:
{
lean_object* v___f_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___f_10_ = ((lean_object*)(lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__0));
v___x_11_ = lean_obj_once(&lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1, &lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1_once, _init_lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1);
v___x_12_ = l_List_foldrTR___redArg(v___f_10_, v___x_11_, v_s_9_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0___redArg(lean_object* v_s_13_, lean_object* v_f_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_15_ = lp_mathlib_Multiset_map___redArg(v_f_14_, v_s_13_);
v___x_16_ = lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0(v___x_15_);
return v___x_16_;
}
}
static lean_object* _init_lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___closed__0(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lean_unsigned_to_nat(3u);
v___x_18_ = l_List_finRange(v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq(lean_object* v_k_19_){
_start:
{
lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___f_20_ = lean_alloc_closure((void*)(lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___lam__0), 2, 1);
lean_closure_set(v___f_20_, 0, v_k_19_);
v___x_21_ = lean_obj_once(&lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___closed__0, &lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___closed__0_once, _init_lp_HoloEngine_MechanicaFluidorum_FourierZ3_k__sq___closed__0);
v___x_22_ = lp_HoloEngine_Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0___redArg(v___x_21_, v___f_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0(lean_object* v_00_u03b9_23_, lean_object* v_s_24_, lean_object* v_f_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_HoloEngine_Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0___redArg(v_s_24_, v_f_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_zero__mode(lean_object* v_x_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_obj_once(&lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1, &lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1_once, _init_lp_HoloEngine_Multiset_sum___at___00Finset_sum___at___00MechanicaFluidorum_FourierZ3_k__sq_spec__0_spec__0___closed__1);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_zero__mode___boxed(lean_object* v_x_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_HoloEngine_MechanicaFluidorum_FourierZ3_zero__mode(v_x_29_);
lean_dec(v_x_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_HoloEngine_MechanicaFluidorum_FourierZ3_negWavevector(lean_object* v_k_31_, lean_object* v_i_32_){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_apply_1(v_k_31_, v_i_32_);
v___x_34_ = lean_int_neg(v___x_33_);
lean_dec(v___x_33_);
return v___x_34_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Complex_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Real_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(uint8_t builtin);
void lean_initialize();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_HoloEngine_HoloEngine_FourierStateZ3(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Data_Complex_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Real_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Ring_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
