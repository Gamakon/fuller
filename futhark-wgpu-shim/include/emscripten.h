/* A native stand-in for the one Emscripten function Futhark's webgpu runtime
 * calls (surface/emscripten_symbols.txt): emscripten_sleep(0), the yield in
 * the runtime's spin-until-the-callback-fires loops (the default wait path).
 *
 * It is NOT a real Emscripten header. The shim completes every
 * AllowSpontaneous request before the call that started it returns, so the
 * loops it sits in are never entered; the shim counts the calls so that a
 * test can show that (README.md, "The wait paths"). */
#ifndef FUTHARK_WGPU_SHIM_EMSCRIPTEN_H
#define FUTHARK_WGPU_SHIM_EMSCRIPTEN_H
#ifdef __cplusplus
extern "C" {
#endif
void emscripten_sleep(unsigned int ms);
#ifdef __cplusplus
}
#endif
#endif
