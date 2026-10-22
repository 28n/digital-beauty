# suminagashi

A single-page digital artwork: living marbled ink on warm washi paper.
A GPU fluid simulation (stable fluids: semi-Lagrangian advection, Jacobi
pressure projection, vorticity confinement) drives three pigments —
indigo, madder, ochre — advected in a dye buffer. A choreography layer
schedules invisible "hands", pigment drops, vortices, calligraphic brush
strokes, and comb passes that shape and occasionally erase the print.

## Run

Any static file server works — there is no build step and no dependency:

```
python3 -m http.server 8123
# open http://localhost:8123/
```

## Files

- `index.html` — fullscreen canvas, nothing else
- `main.js` — WebGL2 sim, shaders, choreography, pointer interaction

## Notes

- Respects `prefers-reduced-motion` by rendering a single still print.
- Pigments live in separate RGB channels of the dye buffer; the display
  shader mixes them onto paper with grain, relief lighting, and vignette.
- If pigment coverage saturates or empties over time, tune
  `DYE_DISSIPATION` and hand `dyeRate` in `main.js`.
