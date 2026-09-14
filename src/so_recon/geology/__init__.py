"""The latent geological measure of E02: its conditioning, its density and its renderer.

E01 defined a twelve-coefficient forward generator and claimed no density over it. This
package supplies the missing law — `p(theta | G)` for the sparse `log k` of the static
observations — and the deterministic map from a latent point back to physical coefficient
arrays.

Three boundaries hold the package together:

* `conditional` owns the algebra and the `PriorContext` it produces. It builds its design
  matrix out of E01's own kernel, so the basis cannot drift from the generator's.
* `density` owns the normalised measure: standard normals in whitened coordinates and a
  fair coin over the two `kz/kx` hypotheses. Nothing here is truncated to the region that
  renders nicely, and no renderer Jacobian is folded in (plan §2's `measure`).
* `renderer` owns the latent-to-arrays map. It reads no truth and writes no file; a
  geology E01 refuses to publish is a `RendererNumericalError`, never a zero density.
"""
