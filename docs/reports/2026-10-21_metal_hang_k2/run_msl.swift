import Metal
import Foundation
let a = CommandLine.arguments
var src = try! String(contentsOfFile: a[1])
let secs = a.count > 2 ? Double(a[2])! : 10
// naga's fake bindings become real buffer slots in order of appearance
var n = 0
while let r = src.range(of: "[[user(fake0)]]") {
    src.replaceSubrange(r, with: "[[buffer(\(n))]]"); n += 1
}
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions()
opts.languageVersion = .version2_4
let lib: MTLLibrary
do { lib = try dev.makeLibrary(source: src, options: opts) } catch { print("COMPILE ERROR \(error)"); exit(2) }
let fn = lib.makeFunction(name: "main_")!
let pso = try! dev.makeComputePipelineState(function: fn)
let q = dev.makeCommandQueue()!
let h = dev.makeBuffer(length: 16 * 16 * 4, options: .storageModeShared)!
var cfg: [UInt32] = [16, 4, 16, 0]
let c = dev.makeBuffer(bytes: &cfg, length: 16, options: .storageModeShared)!
var sizes: [UInt32] = [UInt32(16 * 16 * 4), 0, 0, 0]
let s = dev.makeBuffer(bytes: &sizes, length: 16, options: .storageModeShared)!
let cb = q.makeCommandBuffer()!
let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pso)
e.setBuffer(h, offset: 0, index: 0); e.setBuffer(c, offset: 0, index: 1); e.setBuffer(s, offset: 0, index: 2)
e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
e.endEncoding()
let sem = DispatchSemaphore(value: 0)
cb.addCompletedHandler { _ in sem.signal() }
cb.commit()
let t = Date()
if sem.wait(timeout: .now() + secs) == .timedOut { print("HANG after \(secs) s"); exit(3) }
print("OK in \(Date().timeIntervalSince(t) * 1000) ms status \(cb.status.rawValue) error \(String(describing: cb.error))")
